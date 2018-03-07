#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/slicing_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SlicingForward(const int top_count, const Dtype* guidance_data, const Dtype* grid_data, Dtype* out, 
                               const int coefficient_len, const int img_height , const int img_width, const int grid_channels, const int offset_x, const int offset_y, const int offset_z,
                               const Dtype scale_x, const Dtype scale_y, const int depth_d, const int grid_height, const int grid_width) {
  CUDA_KERNEL_LOOP(index, top_count) {
    // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int coeff_ind = (index / (img_width * img_height)) % coefficient_len;
    int n = index / (img_width * img_height * coefficient_len);

    // Assume guidance is a single channels image.
    int guidance_offset = n * img_width * img_height + y * img_width + x; 

    int x_offset = scale_x * x + offset_x;
    int y_offset = scale_y * y + offset_y;
    int z_offset = depth_d * guidance_data[ guidance_offset] + offset_z;
    Dtype x_distance,y_distance, z_distance;
    
    for (int i = x_offset ; i <= x_offset + 1; ++i){
      if (i < grid_width) {
         x_distance = 1 - ((scale_x * x + offset_x - i ) > 0.0 ? (scale_x * x + offset_x - i ) : -(scale_x * x + offset_x - i ));

        for (int j = y_offset ; j <= y_offset + 1; ++j){
          if (j < grid_height){
             y_distance = 1 - ((scale_y * y + offset_y - j ) > 0.0 ? (scale_y * y + offset_y- j) : -(scale_y * y + offset_y- j ));
             
            for (int k = z_offset; k <= z_offset + 1; ++k){
              if ( k < grid_channels ) {
                Dtype z_abs = (depth_d * guidance_data[ guidance_offset] + offset_z - k );
                z_distance =  1 - ( z_abs > 0.0 ? z_abs : -z_abs );

                Dtype scale =  x_distance * y_distance * z_distance;
                
                int grid_offset = ((n * (coefficient_len * grid_channels) + (k * coefficient_len + coeff_ind)) * grid_height + j) * grid_width + i;

                  out[index] += scale * grid_data[grid_offset];


              }
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void SlicingGuidanceBackward(const int count, const Dtype* guidance_data, const Dtype* grid_data,
                                Dtype* guidance_diff, const Dtype *top_diff,
                               const int coefficient_len, const int img_height , const int img_width, const int grid_channels, const int offset_x, const int offset_y, const int offset_z,
                               const Dtype scale_x, const Dtype scale_y, const int depth_d, const int grid_height, const int grid_width) {
  CUDA_KERNEL_LOOP(index, count) {
    // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);
    int guidance_offset = index; 
    int x_offset = scale_x * x + offset_x;
    int y_offset = scale_y * y + offset_y;
    int z_offset = depth_d * guidance_data[ guidance_offset] + offset_z;
    Dtype x_distance,y_distance;
    Dtype value = 0.0f;
    for (int i = x_offset ; i <= x_offset + 1; ++i){
      if (i < grid_width) {
         x_distance = 1.0 - ((scale_x * x + offset_x - i ) > 0.0 ? (scale_x * x + offset_x - i ) : -(scale_x * x + offset_x - i ));
         
        for (int j = y_offset ; j <= y_offset + 1; ++j){
          if (j < grid_height){
             y_distance = 1.0 - ((scale_y * y + offset_y - j ) > 0.0 ? (scale_y * y + offset_y - j) : -(scale_y * y + offset_y - j ));
             
            for (int k = z_offset; k <= z_offset + 1; ++k){
              if ( k < grid_channels ) {
                Dtype z_abs = (depth_d * guidance_data[ guidance_offset] + offset_z - k );
                
                // w.r.t guidance map
                Dtype scale2 = x_distance * y_distance ;
                for(int coeff_ind = 0; coeff_ind  < coefficient_len; ++coeff_ind){
                 
                  Dtype tao_diff = z_abs > 0.0 ? -1 * depth_d: depth_d; 
                  value += top_diff[((n * coefficient_len + coeff_ind) * img_height + y) * img_width + x]  
                                                            * scale2 * grid_data[((n * (coefficient_len * grid_channels) + (k * coefficient_len + coeff_ind)) * grid_height + j) * grid_width + i] 
                                                            * tao_diff;                                          
                }
                
                
              }
            }
          }
        }
      }
    }
    guidance_diff[guidance_offset] = value;
  }
}

template <typename Dtype>
__global__ void SlicingGridBackward(const int grid_count, const Dtype* guidance_data, const Dtype* grid_data,
                                 Dtype* grid_diff, const Dtype *top_diff,
                               const int coefficient_len, const int img_height , const int img_width, const int grid_channels, const int offset_x, const int offset_y, const int offset_z,
                               const Dtype scale_x, const Dtype scale_y, const int depth_d, const int grid_height, const int grid_width) {
  CUDA_KERNEL_LOOP(index, grid_count) {
    int gx = index % grid_width;
    int gy = (index / grid_width) % grid_height;
    int c_idx = (index / (grid_width * grid_height)) % (grid_channels * coefficient_len);
    int gz = c_idx / coefficient_len;
    if(gz < offset_z || gx < offset_x || gy < offset_y){
      continue;
    }
    int coeff_ind = c_idx % coefficient_len;
    int n = (index / (grid_width * grid_height * grid_channels * coefficient_len));
    Dtype scalar_x = 1.0 / scale_x;
    Dtype scalar_y = 1.0 / scale_y;
    int left_x = static_cast<int> (floor(scalar_x * (gx - offset_x - 1)));
    int right_x = static_cast<int> (ceil(scalar_x * (gx - offset_x + 1)));
    int left_y = static_cast<int> (floor(scalar_y * (gy - offset_y - 1)));
    int right_y = static_cast<int> (ceil(scalar_y * (gy - offset_y + 1)));

    Dtype value = 0.0f;
    Dtype x_distance,y_distance, z_distance;
    // for(int x = 0; x < img_width; ++x){
    for (int x = left_x; x < right_x; ++x){
      if ( x < 0 || x >= img_width){
          continue;
      }
      Dtype gx2 = x * scale_x + offset_x;
      if (gx2 >= grid_width){
        continue;
      }
      x_distance = max(1.0f-abs(gx2 - gx), 0.0f);
      
      // for (int y = 0; y < img_height; ++y){
      for (int y = left_y; y < right_y; ++y){
        if (y < 0 || y >= img_height){
          continue;
        }
        Dtype gy2 = y * scale_y + offset_y;
        if (gy2 >= grid_height){
          continue;
        }

        y_distance = max(1.0f - abs(gy2 - gy), 0.0f);
        int guidance_offset = n * img_height * img_width + y * img_width + x;
        Dtype gz2 = depth_d * guidance_data[guidance_offset] + offset_z;
        if(gz2 >= grid_channels){
          continue;
        }
        
        z_distance = max(1.0f-abs((gz2 - gz )), 0.0f);
        int back_idx = n * coefficient_len * img_height * img_width 
                        + coeff_ind * img_height * img_width + y * img_width + x;
        value += x_distance * y_distance * z_distance * top_diff[back_idx];

      }
    }
    grid_diff[index] = value;
  }
}


template <typename Dtype>
void SlicingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  scale_x_         = this->layer_param_.slicing_param().scale_x();
  scale_y_         = this->layer_param_.slicing_param().scale_y();
  depth_d_         = this->layer_param_.slicing_param().depth_d();

  if (depth_d_ < 0){
    depth_d_ = (bottom[1] -> channels() / coefficient_len_ - 1);
  }
  CHECK(coefficient_len_ * depth_d_ <= bottom[1] -> channels());
  if(scale_x_ < 0){
    scale_x_ = (bottom[1] -> width() - 1)  / (Dtype)bottom[0] -> width();  
  }
  if (scale_y_ < 0){
    scale_y_ = (bottom[1] -> height() - 1) / (Dtype)bottom[0] -> height();
  }
  const Dtype *guidance_data = bottom[0] -> gpu_data();
  const Dtype *grid_data = bottom[1] -> gpu_data();
  Dtype *top_data = top[0] -> mutable_gpu_data();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int grid_height = bottom[1] -> height();
  const int grid_width = bottom[1] -> width();
  const int N = bottom[0] -> num();
  caffe_gpu_set<Dtype>(top[0] -> count(), 0.0, top_data);
  CHECK_EQ(bottom[0] -> channels(), 1);
  
  CHECK(bottom[1] -> channels() % coefficient_len_ == 0);
  const int grid_channels = bottom[1] -> channels() / coefficient_len_;
  

  const int top_count = top[0] -> count();
  SlicingForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
       top_count,  guidance_data,  grid_data, top_data, 
                                coefficient_len_,  img_height, img_width, grid_channels, offset_x_, offset_y_, offset_z_, 
                                scale_x_, scale_y_,  depth_d_,  grid_height, grid_width);
  CUDA_POST_KERNEL_CHECK;
  
}

template <typename Dtype>
void SlicingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


  Dtype *guidance_diff = bottom[0] -> mutable_gpu_diff();
  Dtype *grid_diff = bottom[1] -> mutable_gpu_diff();
  const Dtype *grid_data = bottom[1] -> gpu_data();
  const Dtype *guidance_data = bottom[0] -> gpu_data();

  const Dtype *top_diff = top[0] -> gpu_diff();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int grid_height = bottom[1] -> height();
  const int grid_width = bottom[1] -> width();
  const int N = bottom[0] -> num();
  caffe_gpu_set<Dtype>(bottom[0] -> count(), 0.0, guidance_diff);
  caffe_gpu_set<Dtype>(bottom[1] -> count(), 0.0, grid_diff);
  
  const int count = bottom[0] -> count();
  const int grid_channels = bottom[1] -> channels() / coefficient_len_;
  

  SlicingGuidanceBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, guidance_data, grid_data,
                                guidance_diff, top_diff,
                                coefficient_len_,img_height, img_width, grid_channels, offset_x_, offset_y_, offset_z_,
                                scale_x_, scale_y_,  depth_d_,  grid_height, grid_width
                                );
  CUDA_POST_KERNEL_CHECK;

  const int grid_count = bottom[1] -> count();
  SlicingGridBackward<Dtype><<<CAFFE_GET_BLOCKS(grid_count), CAFFE_CUDA_NUM_THREADS>>>(
    grid_count, guidance_data, grid_data
                                , grid_diff, top_diff,
                                coefficient_len_,img_height, img_width, grid_channels, offset_x_, offset_y_, offset_z_,
                                scale_x_, scale_y_,  depth_d_,  grid_height, grid_width
                                );
  CUDA_POST_KERNEL_CHECK;



}

INSTANTIATE_LAYER_GPU_FUNCS(SlicingLayer);

}  // namespace caffe
