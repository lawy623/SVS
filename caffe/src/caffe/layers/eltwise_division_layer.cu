#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_division_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void epsilon_div_kernel_forward(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, Dtype epsilon) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(b[index]) < epsilon ? 0 : a[index] / b[index];
  }
}


template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0] -> count();
  const Dtype *a_value = bottom[0] -> gpu_data();
  const Dtype *b_value = bottom[1] -> gpu_data();
  Dtype *top_value = top[0] -> mutable_gpu_data();

  epsilon_div_kernel_forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       count,  a_value,  b_value, top_value, epsilon_);
}

template <typename Dtype>
__global__ void epsilon_div_kernel_backward(const int n, const Dtype* a_value,
    const Dtype* b_value, const Dtype *top_diff, Dtype* a_diff, Dtype* b_diff, Dtype epsilon) {
  CUDA_KERNEL_LOOP(index, n) {
    if (abs(b_value[index]) < epsilon){
      a_diff[index] = 0.0;
      b_diff[index] = 0.0;
    }else{
      a_diff[index] = top_diff[index] / b_value[index];
      b_diff[index] = top_diff[index] * (-a_value[index]) / (b_value[index]*b_value[index]);
    }
  }
}
template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0] -> count();
  const Dtype *a_value = bottom[0] -> gpu_data();
  const Dtype *b_value = bottom[1] -> gpu_data();
  const Dtype *top_diff = top[0] -> gpu_diff();
  Dtype *a_diff = bottom[0] -> mutable_gpu_diff();
  Dtype *b_diff = bottom[1] -> mutable_gpu_diff();

  epsilon_div_kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       count,  a_value,  b_value, top_diff, a_diff, b_diff, epsilon_);
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseDivisionLayer);

}  // namespace caffe
