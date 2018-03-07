#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bilateral_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "math.h"


namespace caffe {

template <typename Dtype>
__device__ Dtype diff_abs(Dtype x) {
  Dtype eps = 1e-8;
  return sqrt(x*x+eps);
}
template <typename Dtype>
__device__ Dtype d_diff_abs(Dtype x) {
  Dtype eps = 1e-8;
  return x/sqrt(x*x+eps);
}
template <typename Dtype>
__device__ Dtype weight_z(Dtype x) {
  Dtype abx = diff_abs(x);
  return max(1.0f-abx, 0.0f);
}
template <typename Dtype>
__device__ Dtype d_weight_z(Dtype x) {
  Dtype abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}
template <typename Dtype>
__global__ void BilateralSliceKernel(
    int nthreads,
    const Dtype* grid, const Dtype* guide,
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd, const Dtype scale_x, const Dtype scale_y,const int offset_x, const int offset_y, const int offset_z,
    Dtype* out)
{
  // - Samples centered at 0.5.
  // - Repeating boundary conditions

  CUDA_KERNEL_LOOP(index, nthreads) {

    int x = index % w;
    int y = (index / w) % h;
    int c = index / (w * h) % chans;
    int b = index / (chans * w * h);

    Dtype gx = (x+0.5f) * scale_x;
    Dtype gy = (y+0.5f) * scale_y;
    
    Dtype gz = guide[x + w*(y + h*b)] * gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // int sz = chans;
    // int sx = chans*gd;
    // int sy = chans*gd*gw;
    // int sb = chans*gd*gw*gh;

    Dtype value = 0.0f;
    for (int xx = fx; xx < fx+2; ++xx) {
      
      int x_ = max(min(xx, gw-1), 0);
      Dtype wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
      for (int yy = fy; yy < fy+2; ++yy)
      {
      
        int y_ = max(min(yy, gh-1), 0);
        Dtype wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
        for (int zz = fz; zz < fz+2; ++zz)
        {
      
          int z_ = max(min(zz, gd-1), 0);
          Dtype wz = weight_z(zz+0.5-gz);
          int grid_idx =  ((b * (chans * gd) + (z_ * chans + c)) * gh + y_) * gw + x_;
          value += grid[grid_idx]*wx*wy*wz;
        }
      }

    }
    out[index] = value;
  }
}



  //   Args:
//     
//     guide: (Tensor) [batch_size, h, w ] guide map to slice along.
//     name: (string) name for the operation.
// grid: (Tensor) [batch_size, grid_h, grid_w, depth * n_outputs]
//       grid to slice from.
//   Returns:
//     sliced: (Tensor) [batch_size, h, w, n_outputs] sliced output.
//   """


template <typename Dtype>
void BilateralSlicingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  scale_x_         = this->layer_param_.slicing_param().scale_x();
  scale_y_         = this->layer_param_.slicing_param().scale_y();
  depth_d_         = this->layer_param_.slicing_param().depth_d();

  if (depth_d_ < 0){
    depth_d_ = (bottom[1] -> channels() / coefficient_len_ );
  }
  CHECK(coefficient_len_ * depth_d_ <= bottom[1] -> channels());
  if(scale_x_ < 0){
    scale_x_ = (bottom[1] -> width() )  / (Dtype)bottom[0] -> width();  
  }
  if (scale_y_ < 0){
    scale_y_ = (bottom[1] -> height() ) / (Dtype)bottom[0] -> height();
  }

  const Dtype *guide = bottom[0] -> gpu_data();
  const Dtype *grid = bottom[1] -> gpu_data();
  Dtype *out = top[0] -> mutable_gpu_data();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int grid_height = bottom[1] -> height();
  const int grid_width = bottom[1] -> width();
  const int N = bottom[0] -> num();
  caffe_gpu_set<Dtype>(top[0] -> count(), 0.0, out);
  CHECK_EQ(bottom[0] -> channels(), 1);
  
  CHECK(bottom[1] -> channels() % coefficient_len_ == 0);

  
  int bs = N;
  int h = img_height;
  int w = img_width;
  int gh = grid_height;
  int gw = grid_width;
  int gd = depth_d_;
  int chans = coefficient_len_;

  int total_count = bs*chans*h*w;
  if (total_count > 0) {
    BilateralSliceKernel<Dtype><<<CAFFE_GET_BLOCKS(total_count), CAFFE_CUDA_NUM_THREADS>>>(
        total_count, grid, guide,
        bs, h, w, chans, gh, gw, gd, scale_x_, scale_y_, offset_x_, offset_y_, offset_z_, 
        out);
  }

  
  CUDA_POST_KERNEL_CHECK;
  
}




template <typename Dtype>
__global__ void BilateralSliceGridGradKernel(
    int nthreads,
    const Dtype* grid, const Dtype* guide, const Dtype* backprop,
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd, const Dtype scale_x, const Dtype scale_y,const int offset_x, const int offset_y, const int offset_z,
    Dtype* out)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // int c = idx % chans;
    // int gz = (idx / chans) % gd;
    // int gx = (idx / (chans*gd)) % gw;
    // int gy = (idx / (chans*gd*gw)) % gh;
    // int b = (idx / (chans*gd*gw*gh));
    int gx = idx % gw;
    int gy = (idx / gw) % gh;
    int c_idx = (idx / (gw * gh)) % (chans * gd);
    int b = (idx / (gw * gh * gd * chans));
    int c = c_idx % chans;
    int gz = c_idx / chans;

    
    Dtype scale_w = 1.0 / scale_x;
    Dtype scale_h = 1.0 / scale_y;

    int left_x = static_cast<int>(floor(scale_w*(gx+0.5-1)));
    int right_x = static_cast<int>(ceil(scale_w*(gx+0.5+1)));
    int left_y = static_cast<int>(floor(scale_h*(gy+0.5-1)));
    int right_y = static_cast<int>(ceil(scale_h*(gy+0.5+1)));
    Dtype value = 0.0f;
   
    for (int x = left_x; x < right_x; ++x)
    {
      int x_ = x;

      // mirror boundary
      if (x_ < 0) x_ = -x_-1;
      if (x_ >= w) x_ = 2*w-1-x_;

      
      Dtype gx2 = (x+0.5f)/scale_w;
      Dtype wx = max(1.0f-abs(gx+0.5-gx2), 0.0f);

      for (int y = left_y; y < right_y; ++y)
      {
        int y_ = y;

        // mirror boundary
        if (y_ < 0) y_ = -y_-1;
        if (y_ >= h) y_ = 2*h-1-y_;

        

        Dtype gy2 = (y+0.5f)/scale_h;
        Dtype wy = max(1.0f-abs(gy+0.5-gy2), 0.0f);

        int guide_idx = x_ + w*y_ + h*w*b;
        Dtype gz2 = guide[guide_idx]*gd;
        
        Dtype wz = weight_z(gz+0.5f-gz2);
        if ((gz==0 && gz2<0.5f) || (gz==gd-1 && gz2>gd-0.5f)) {
          wz = 1.0f;
        }

        int back_idx = ((b * chans + c) * h + y_) * w + x_;
        value += wz*wx*wy*backprop[back_idx];
      }
    }
    
    out[idx] = value;
  }
}



template <typename Dtype>
__global__ void BilateralSliceGuideGradKernel(
    int nthreads,
    const Dtype* grid, const Dtype* guide, const Dtype* backprop,
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd, const Dtype scale_x, const Dtype scale_y,const int offset_x, const int offset_y, const int offset_z,
    Dtype* out)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    int x = idx  % w;
    int y = (idx / w) % h;
    int b = (idx / (w*h));

    Dtype gx = (x+0.5f)*scale_x;
    Dtype gy = (y+0.5f)*scale_y;
    Dtype gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));


    Dtype value = 0.0f;
    for (int c = 0; c < chans; ++c) {
      Dtype chan_val = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        if (xx >= gw) continue;
        int x_ = max(min(xx, gw-1), 0);
        Dtype wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          if (yy >= gh) continue;
          int y_ = max(min(yy, gh-1), 0);
          Dtype wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {
            if (zz >= gd) continue;
            int z_ = max(min(zz, gd-1), 0);
            Dtype dwz = gd*d_weight_z(zz+0.5-gz);

            
            int grid_idx =  ((b * (chans * gd) + (z_ * chans + c)) * gh + y_) * gw + x_;
            chan_val += grid[grid_idx]*wx*wy*dwz;
          }
        }
      }
      int back_idx = ((b * chans + c) * h + y) * w + x;
      chan_val *= backprop[back_idx];
      value += chan_val;
    }
    out[idx] = value;
  }
}

template <typename Dtype>
void BilateralSlicingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


  Dtype *guidance_diff = bottom[0] -> mutable_gpu_diff();
  Dtype *grid_diff = bottom[1] -> mutable_gpu_diff();
  const Dtype *grid = bottom[1] -> gpu_data();
  const Dtype *guide = bottom[0] -> gpu_data();

  const Dtype *backprop = top[0] -> gpu_diff();

  int bs = bottom[1] -> num();
  int gh = bottom[1] -> height();
  int gw = bottom[1] -> width();
  int gd = depth_d_;
  int chans = coefficient_len_;

  int h = bottom[0] -> height();
  int w = bottom[0] -> width();
  int grid_count = bs*gh*gw*gd*chans;
  caffe_gpu_set<Dtype>(bottom[0] -> count(), 0.0, guidance_diff);
  caffe_gpu_set<Dtype>(bottom[1] -> count(), 0.0, grid_diff);

  if (grid_count > 0) {
    
    BilateralSliceGridGradKernel<Dtype><<<CAFFE_GET_BLOCKS(grid_count), CAFFE_CUDA_NUM_THREADS>>>(
        grid_count, grid, guide, backprop,
        bs, h, w, chans, gh, gw, gd,scale_x_, scale_y_,offset_x_, offset_y_, offset_z_, 
        grid_diff);
  }

  int guide_count = bs*h*w;
  if (guide_count > 0) {
    
    BilateralSliceGuideGradKernel<Dtype><<<CAFFE_GET_BLOCKS(guide_count), CAFFE_CUDA_NUM_THREADS>>>(
        guide_count, grid, guide, backprop,
        bs, h, w, chans, gh, gw, gd, scale_x_, scale_y_,offset_x_, offset_y_, offset_z_, 
        guidance_diff);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BilateralSlicingLayer);

}  // namespace caffe
