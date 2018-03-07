#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <boost/thread.hpp>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/stereo_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "thirdparty/CImg/CImg.h"
using namespace cimg_library;

namespace caffe {

cv::Mat ReadPFMToCVMat(const string& filename,
                       const int height,
                       const int width)
{
  CImg<float> pfm_img;
  pfm_img.load_pfm(filename.c_str());
  if (width != 0)
    CHECK_EQ(pfm_img.width(), width) << "Incorrect image width";
  int pfm_width = pfm_img.width();
  if (height != 0)
    CHECK_EQ(pfm_img.height(), height) << "Incorrect image height";
  int pfm_height = pfm_img.height();
  cv::Mat cv_img(cv::Size(pfm_width, pfm_height), CV_32FC1);
  cimg_forXY(pfm_img, x, y)
  {
     cv_img.at<float>(y, x, 0) = -1.f * pfm_img(x, y, 0, 0);
  }
  return cv_img;
}

template <typename Dtype>
StereoImageDataLayer<Dtype>::StereoImageDataLayer(const LayerParameter& param)
   : BaseDataLayer<Dtype>(param),
    prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
} 


template <typename Dtype>
StereoImageDataLayer<Dtype>::~StereoImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void StereoImageDataLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  this->output_labels_ = false;
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_l_.mutable_cpu_data();
    prefetch_[i].data_r_.mutable_cpu_data();
    prefetch_[i].data_gt_.mutable_cpu_data();
    if(has_mask_)
      prefetch_[i].data_m_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_l_.mutable_gpu_data();
      prefetch_[i].data_r_.mutable_gpu_data();
      prefetch_[i].data_gt_.mutable_gpu_data();
      if (has_mask_)
        prefetch_[i].data_m_.mutable_gpu_data();
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void StereoImageDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif
  try {
    while (!must_stop()) {
      StereoBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_l_.data().get()->async_gpu_push(stream);
        batch->data_r_.data().get()->async_gpu_push(stream);
        batch->data_gt_.data().get()->async_gpu_push(stream);
        if(has_mask_)
          batch->data_m_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
   //INTERRUP
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void StereoImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  has_mask_  = this->layer_param_.image_data_param().has_mask();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename_l;
  string filename_r;
  string filename_gt;
  string filename_m;
  if (has_mask_) {
    while (infile >> filename_l >> filename_r >> filename_gt >> filename_m) {
      lines_.push_back(std::make_pair(std::make_pair(filename_l, filename_r), filename_gt));
      lines_m_.push_back(filename_m);
    }
  }
  else {
    while (infile >> filename_l >> filename_r >> filename_gt) {
      lines_.push_back(std::make_pair(std::make_pair(filename_l, filename_r), filename_gt));
    }
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img_l = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                    new_height, new_width, is_color);
  cv::Mat cv_img_r = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                    new_height, new_width, is_color);
  cv::Mat cv_img_gt = ReadPFMToCVMat(root_folder + lines_[lines_id_].second,
                                    new_height, new_width);
  cv::Mat cv_img_m;
  if (has_mask_)
    cv_img_m = ReadImageToCVMat(root_folder + lines_m_[lines_id_],
                                    new_height, new_width);
  CHECK(cv_img_l.data) << "Could not load " << root_folder << lines_[lines_id_].first.first;
  CHECK(cv_img_r.data) << "Could not load " << root_folder << lines_[lines_id_].first.second;
  CHECK(cv_img_gt.data) << "Could not load " << root_folder << lines_[lines_id_].second;
  if (has_mask_)
    CHECK(cv_img_m.data) << "Could not load " << root_folder << lines_m_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape_l = this->data_transformer_->InferBlobShape(cv_img_l);
  vector<int> top_shape_r = this->data_transformer_->InferBlobShape(cv_img_r);
  vector<int> top_shape_gt = this->data_transformer_->InferBlobShape(cv_img_gt);
  vector<int> top_shape_m;
  if (has_mask_)
    top_shape_m = this->data_transformer_->InferBlobShape(cv_img_m);
  this->transformed_data_l_.Reshape(top_shape_l);
  this->transformed_data_r_.Reshape(top_shape_r);
  this->transformed_data_gt_.Reshape(top_shape_gt);
  if (has_mask_)
    this->transformed_data_m_.Reshape(top_shape_m);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape_l[0] = batch_size;
  top_shape_r[0] = batch_size;
  top_shape_gt[0] = batch_size;
  if (has_mask_)
    top_shape_m[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_l_.Reshape(top_shape_l);
    this->prefetch_[i].data_r_.Reshape(top_shape_r);
    this->prefetch_[i].data_gt_.Reshape(top_shape_gt);
    if (has_mask_)
      this->prefetch_[i].data_m_.Reshape(top_shape_m);
  }
  top[0]->Reshape(top_shape_l);
  top[1]->Reshape(top_shape_r);
  top[2]->Reshape(top_shape_gt);
  if (has_mask_)
    top[3]->Reshape(top_shape_m);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void StereoImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void StereoImageDataLayer<Dtype>::load_batch(StereoBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_l_.count());
  CHECK(batch->data_r_.count());
  CHECK(batch->data_gt_.count());
  CHECK(this->transformed_data_l_.count());
  CHECK(this->transformed_data_r_.count());
  CHECK(this->transformed_data_gt_.count());
  if (has_mask_) {
    CHECK(batch->data_m_.count());
    CHECK(this->transformed_data_m_.count());
  }
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img_l = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                    new_height, new_width, is_color);
  cv::Mat cv_img_r = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                    new_height, new_width, is_color);
  cv::Mat cv_img_gt = ReadPFMToCVMat(root_folder + lines_[lines_id_].second,
                                    new_height, new_width);
  cv::Mat cv_img_m;
  if (has_mask_)
    cv_img_m = ReadImageToCVMat(root_folder + lines_m_[lines_id_],
                                    new_height, new_width);
  CHECK(cv_img_l.data) << "Could not load " << root_folder << lines_[lines_id_].first.first;
  CHECK(cv_img_r.data) << "Could not load " << root_folder << lines_[lines_id_].first.second;
  CHECK(cv_img_gt.data) << "Could not load " << root_folder << lines_[lines_id_].second;
  if (has_mask_)
    CHECK(cv_img_m.data) << "Could not load " << root_folder << lines_m_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape_l = this->data_transformer_->InferBlobShape(cv_img_l);
  vector<int> top_shape_r = this->data_transformer_->InferBlobShape(cv_img_r);
  vector<int> top_shape_gt = this->data_transformer_->InferBlobShape(cv_img_gt);
  vector<int> top_shape_m;
  if (has_mask_)
    top_shape_m = this->data_transformer_->InferBlobShape(cv_img_m);
  this->transformed_data_l_.Reshape(top_shape_l);
  this->transformed_data_r_.Reshape(top_shape_r);
  this->transformed_data_gt_.Reshape(top_shape_gt);
  if (has_mask_)
    this->transformed_data_m_.Reshape(top_shape_m);
  // Reshape batch according to the batch_size.
  top_shape_l[0] = batch_size;
  top_shape_r[0] = batch_size;
  top_shape_gt[0] = batch_size;
  if (has_mask_)
    top_shape_m[0] = batch_size;
  batch->data_l_.Reshape(top_shape_l);
  batch->data_r_.Reshape(top_shape_r);
  batch->data_gt_.Reshape(top_shape_gt);
  if (has_mask_)
    batch->data_m_.Reshape(top_shape_m);

  Dtype* prefetch_data_l = batch->data_l_.mutable_cpu_data();
  Dtype* prefetch_data_r = batch->data_r_.mutable_cpu_data();
  Dtype* prefetch_data_gt = batch->data_gt_.mutable_cpu_data();
  Dtype* prefetch_data_m;
  if (has_mask_)
    prefetch_data_m = batch->data_m_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img_l = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                      new_height, new_width, is_color);
    cv::Mat cv_img_r = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                      new_height, new_width, is_color);
    cv::Mat cv_img_gt = ReadPFMToCVMat(root_folder + lines_[lines_id_].second,
                                      new_height, new_width);
    cv::Mat cv_img_m;
    if (has_mask_)
      cv_img_m = ReadImageToCVMat(root_folder + lines_m_[lines_id_],
                                      new_height, new_width);
    CHECK(cv_img_l.data) << "Could not load " << root_folder << lines_[lines_id_].first.first;
    CHECK(cv_img_r.data) << "Could not load " << root_folder << lines_[lines_id_].first.second;
    CHECK(cv_img_gt.data) << "Could not load " << root_folder << lines_[lines_id_].second;
    if (has_mask_)
      CHECK(cv_img_m.data) << "Could not load " << root_folder << lines_m_[lines_id_];
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_l_.offset(item_id);
    this->transformed_data_l_.set_cpu_data(prefetch_data_l + offset);
    this->data_transformer_->Transform(cv_img_l, &(this->transformed_data_l_));
    offset = batch->data_r_.offset(item_id);
    this->transformed_data_r_.set_cpu_data(prefetch_data_r + offset);
    this->data_transformer_->Transform(cv_img_r, &(this->transformed_data_r_));
    offset = batch->data_gt_.offset(item_id);
    this->transformed_data_gt_.set_cpu_data(prefetch_data_gt + offset);
    Dtype* top_ptr = this->transformed_data_gt_.mutable_cpu_data();
    for (size_t y = 0; y < cv_img_gt.rows; ++ y)
      for (size_t x = 0; x < cv_img_gt.cols; ++ x)
        top_ptr[y * cv_img_gt.cols + x] = cv_img_gt.at<float>(y, x , 0);
    if (has_mask_) {
      offset = batch->data_m_.offset(item_id);
      this->transformed_data_m_.set_cpu_data(prefetch_data_m + offset);
      this->data_transformer_->Transform(cv_img_m, &(this->transformed_data_m_));
    }
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void StereoImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  StereoBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  top[0]->ReshapeLike(batch->data_l_);
  top[1]->ReshapeLike(batch->data_r_);
  top[2]->ReshapeLike(batch->data_gt_);
  if (has_mask_)
    top[3]->ReshapeLike(batch->data_m_);
  caffe_copy(batch->data_l_.count(), batch->data_l_.cpu_data(),
             top[0]->mutable_cpu_data());
  caffe_copy(batch->data_r_.count(), batch->data_r_.cpu_data(),
             top[1]->mutable_cpu_data());
  caffe_copy(batch->data_gt_.count(), batch->data_gt_.cpu_data(),
             top[2]->mutable_cpu_data());
  if (has_mask_)
    caffe_copy(batch->data_m_.count(), batch->data_m_.cpu_data(),
               top[3]->mutable_cpu_data());
  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(StereoImageDataLayer, Forward);
#endif

INSTANTIATE_CLASS(StereoImageDataLayer);
REGISTER_LAYER_CLASS(StereoImageData);

}  // namespace caffe
#endif  // USE_OPENCV
