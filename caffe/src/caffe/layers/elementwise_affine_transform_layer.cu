#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/elementwise_affine_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AffineTransformForward(const int count, const Dtype* coeff_data, const Dtype* img_data, Dtype* out,
                                       const int img_height , const int img_width, const int n_theta = 3, const int n_out_channels = 3) {
  CUDA_KERNEL_LOOP(index, count) {
     // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);

    for(int c = 0; c < n_out_channels; c++ ){
      int out_offset = ((n * n_out_channels + c) * img_height + y) * img_width + x;
      int T_offset = ((n * (n_theta  + 1) * n_out_channels + n_theta + (n_theta + 1) * c) * img_height + y) * img_width + x;
      out[out_offset] = coeff_data[T_offset];
      int img_offset = ((n * n_out_channels + 0) * img_height + y) * img_width + x;
      int R_offset = ((n * (n_theta  + 1) * n_out_channels + 0 + (n_theta + 1) * c) * img_height + y) * img_width + x;
      for (int c_head = 0; c_head <= n_theta - 1; ++c_head){
        out[out_offset] += img_data[img_offset] * coeff_data[R_offset];
        img_offset += img_height * img_width;
        R_offset += img_width * img_height;
      }
    }
  }
}

template <typename Dtype>
__global__ void AffineTransformBackward(const int count, Dtype* coeff_diff, Dtype* img_diff, const Dtype* coeff_data, 
                                        const Dtype* img_data,  const Dtype *top_diff,
                                        const int img_height , const int img_width,const int n_theta = 3, const int n_out_channels = 3){
  CUDA_KERNEL_LOOP(index, count) {
     // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);

    for(int c = 0; c < n_out_channels; c++ ){
      int out_offset = ((n * n_out_channels + c) * img_height + y) * img_width + x;
      int T_offset = ((n * (n_theta  + 1) * n_out_channels + n_theta + (n_theta + 1) * c) * img_height + y) * img_width + x;
      coeff_diff[T_offset] = top_diff[out_offset];
      int img_offset = ((n * n_out_channels + 0) * img_height + y) * img_width + x;
      int R_offset = ((n * (n_theta  + 1) * n_out_channels + 0 + (n_theta + 1) * c) * img_height + y) * img_width + x;
      for (int c_head = 0; c_head <= n_theta - 1; ++c_head){
        img_diff[img_offset] += coeff_data[R_offset] * top_diff[out_offset];
        coeff_diff[R_offset] = img_data[img_offset] * top_diff[out_offset];
        img_offset += img_height * img_width;
        R_offset += img_width * img_height;
      }
      
    }
  }
}


template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype *coeff_data = bottom[0] -> gpu_data();
  const Dtype *img_data = bottom[1] -> gpu_data();
  Dtype *top_data = top[0] -> mutable_gpu_data();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int count = bottom[0] -> count() / (bottom[0] -> channels());

  AffineTransformForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       count,  coeff_data,  img_data, top_data, 
       img_height, img_width, n_theta_, n_out_channels_ );

  CUDA_POST_KERNEL_CHECK;
  
}

template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype *coeff_diff = bottom[0] -> mutable_gpu_diff();
  Dtype *img_diff = bottom[1] -> mutable_gpu_diff();
  const Dtype *coeff_data = bottom[0] -> gpu_data();
  const Dtype *img_data = bottom[1] -> gpu_data();

  const Dtype *top_diff = top[0] -> gpu_diff();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
 
  caffe_gpu_set<Dtype>(bottom[0] -> count(), 0.0, coeff_diff);
  caffe_gpu_set<Dtype>(bottom[1] -> count(), 0.0, img_diff);
  
  const int count = bottom[0] -> count() / (bottom[0] -> channels());

  AffineTransformBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
     count,  coeff_diff, img_diff, coeff_data, img_data, top_diff, 
     img_height, img_width, n_theta_, n_out_channels_ );


  CUDA_POST_KERNEL_CHECK;
 
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseAffineTransformLayer);

}  // namespace caffe
