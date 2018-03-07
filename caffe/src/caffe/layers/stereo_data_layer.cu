#include <vector>

#include "caffe/layers/stereo_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void StereoImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  StereoBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  top[0]->ReshapeLike(batch->data_l_);
  top[1]->ReshapeLike(batch->data_r_);
  top[2]->ReshapeLike(batch->data_gt_);
  if (this->has_mask_)
    top[3]->ReshapeLike(batch->data_m_);
  caffe_copy(batch->data_l_.count(), batch->data_l_.gpu_data(),
      top[0]->mutable_gpu_data());
  caffe_copy(batch->data_r_.count(), batch->data_r_.gpu_data(),
      top[1]->mutable_gpu_data());
  caffe_copy(batch->data_gt_.count(), batch->data_gt_.gpu_data(),
      top[2]->mutable_gpu_data());
  if (this->has_mask_)
    caffe_copy(batch->data_m_.count(), batch->data_m_.gpu_data(),
        top[3]->mutable_gpu_data());
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(StereoImageDataLayer);

} // namespace caffe
