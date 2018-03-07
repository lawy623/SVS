// Yang Chengxi added 2016.12.29

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/nn_upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward(const int nthreads, const int b_height, const int b_width,
                        const int t_height, const int t_width, const int resize_,
                        Dtype* top_data, const Dtype* bottom_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int t_w = index % t_width;
    int t_h = (index / t_width) % t_height;
    int cn = (index / t_width) / t_height;
    int b_index = (cn * b_height + t_h / resize_) * b_width + t_w / resize_;
    top_data[index] = bottom_data[b_index];
  }
}

template <typename Dtype>
void NNUpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  int t_count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  int b_width = bottom[0]->width();
  int b_height = bottom[0]->height();
  forward<Dtype><<<CAFFE_GET_BLOCKS(t_count), CAFFE_CUDA_NUM_THREADS>>>(
         t_count, b_height, b_width,
         height_, width_, resize_,
         top_data, bottom_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void backward(const int nthreads, const int b_height, const int b_width,
                         const int t_height, const int t_width, const int resize_,
                         const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = 0;
    int b_w = index % b_width;
    int b_h = (index / b_width) % b_height;
    int cn = (index / b_width) / b_height;
    for (int re_y = 0; re_y < resize_; ++ re_y)
      for (int re_x = 0; re_x < resize_; ++ re_x) {
        int t_index = (cn * t_height + b_h * resize_ + re_y) * t_width + b_w * resize_ + re_x;
        bottom_diff[index] += top_diff[t_index];
      }
  }
}

template <typename Dtype>
void NNUpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int b_count = bottom[0]->count();
  int b_width = bottom[0]->width();
  int b_height = bottom[0]->height();
  const Dtype* top_diff = top[0]->mutable_gpu_diff();
  backward<Dtype><<<CAFFE_GET_BLOCKS(b_count), CAFFE_CUDA_NUM_THREADS>>>(
          b_count, b_height, b_width,
          height_, width_, resize_,
          top_diff, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(NNUpsampleLayer);

}//namespace caffe
