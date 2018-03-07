#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class StereoBatch {
 public:
  Blob<Dtype> data_l_, data_r_, data_gt_, data_m_;
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class StereoImageDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit StereoImageDataLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual ~StereoImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  static const int PREFETCH_COUNT = 2;

  virtual inline const char* type() const { return "StereoImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 3; }
  virtual inline int MaxTopBlobs() const { return 4;}

 protected:
  virtual void InternalThreadEntry();
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(StereoBatch<Dtype>* batch);
  StereoBatch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<StereoBatch<Dtype>*> prefetch_free_;
  BlockingQueue<StereoBatch<Dtype>*> prefetch_full_;
  Blob<Dtype> transformed_data_l_, transformed_data_r_, transformed_data_gt_, transformed_data_m_;

  vector<std::pair<std::pair<std::string, std::string>, std::string> > lines_;
  vector<std::string> lines_m_;
  int lines_id_;
  bool has_mask_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
