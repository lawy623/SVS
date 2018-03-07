// Yang Chengxi added 2016.12.29

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/nn_upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void NNUpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NNUpsampleParameter nn_upsample_param = this->layer_param_.nn_upsample_param(); 
  resize_ = nn_upsample_param.resize();
}

template <typename Dtype>
void NNUpsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height() * resize_;
  width_ = bottom[0]->width() * resize_;
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void NNUpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int b_height = bottom[0]->height();
  const int b_width = bottom[0]->width();
  for (int n = 0; n < num_; ++ n)
    for (int c = 0; c < channels_; ++c)
      for (int h = 0; h < height_; ++ h)
        for (int w = 0; w < width_; ++ w) {
          int top_index = ((n * channels_ + c) * height_ + h) * width_ + w;
          int bottom_index = ((n * channels_ + c) * b_height + h / resize_) * b_width + w / resize_;
          top_data[top_index] = bottom_data[bottom_index];
        }
}

template <typename Dtype>
void NNUpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int b_height = bottom[0]->height();
  const int b_width = bottom[0]->width();
  for (int n = 0; n < num_; ++ n)
    for (int c = 0; c < channels_; ++ c)
      for (int h = 0; h < b_height; ++ h)
        for (int w = 0; w < b_width; ++ w) {
          int bottom_index = ((n * channels_ + c) * b_height + h) * b_width + w;
          bottom_diff[bottom_index] = 0;
          for (int re_y = 0; re_y < resize_; ++ re_y)
            for (int re_x = 0; re_x < resize_; ++ re_x) {
              int top_index = ((n * channels_ + c) * height_ + h * resize_ + re_y) * width_
                                                                      + w * resize_ + re_x;
              bottom_diff[bottom_index] += top_diff[top_index];
            }
        }
}

#ifdef CPU_ONLY
STUB_GPU(NNUpsampleLayer);
#endif

INSTANTIATE_CLASS(NNUpsampleLayer);
REGISTER_LAYER_CLASS(NNUpsample);

}//namespace caffe
