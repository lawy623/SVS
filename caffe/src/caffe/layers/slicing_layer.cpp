#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/slicing_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// bottom[0] is guidance map g, value in range of [0,1], in shape of [N, 1, H,W]
// bottom[1] is bilateral grid, in shape of [N, D * stride, H', w']

template <typename Dtype>
void SlicingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  coefficient_len_ = this->layer_param_.slicing_param().coefficient_len();
  scale_x_         = this->layer_param_.slicing_param().scale_x();
  scale_y_         = this->layer_param_.slicing_param().scale_y();
  depth_d_         = this->layer_param_.slicing_param().depth_d();
  offset_x_         = this->layer_param_.slicing_param().offset_x();
  offset_y_         = this->layer_param_.slicing_param().offset_y();
  offset_z_         = this->layer_param_.slicing_param().offset_z();


  CHECK(bottom[1] -> channels() % coefficient_len_ == 0) << "bilateral grid channels must be divied by coefficient_len_";
  

  
}

template <typename Dtype>
void SlicingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0] -> Reshape(bottom[0] -> num(), coefficient_len_, bottom[0] -> height(), bottom[0] -> width());
}


template <typename Dtype>
void SlicingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

  const Dtype *guidance_data = bottom[0] -> cpu_data();
  const Dtype *grid_data = bottom[1] -> cpu_data();
  Dtype *top_data = top[0] -> mutable_cpu_data();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int grid_height = bottom[1] -> height();
  const int grid_width = bottom[1] -> width();
  // const int N = bottom[0] -> num();
  caffe_set<Dtype>(top[0] -> count(), 0.0, top_data);
  Dtype x_distance,y_distance, z_distance;
  int  guidance_offset;
  CHECK_EQ(bottom[0] -> channels(), 1);
  const int count = bottom[0] -> count();
  CHECK(bottom[1] -> channels() % coefficient_len_ == 0);
  const int grid_channels = bottom[1] -> channels() / coefficient_len_;

  for(int index = 0; index <count; ++index){
  // for (int n = 0; n < N; ++n){
  //   for(int x = 0; x < img_width; ++x){
  //     for(int y = 0; y < img_height; ++y){
    // guidance_offset = bottom[0] -> offset(n, 0, y, x);

    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);
      guidance_offset = index;    
        
        int x_offset = scale_x_ * x + offset_x_;
        int y_offset = scale_y_ * y + offset_y_;
        int z_offset = depth_d_ * guidance_data[ guidance_offset] + offset_z_;

        for (int i = x_offset ; i <= x_offset + 1; ++i){
          if (i < grid_width) {
             x_distance = 1 - ((scale_x_ * x + offset_x_ - i ) > 0.0 ? (scale_x_ * x + offset_x_ - i ) : -(scale_x_ * x + offset_x_ - i ));

            for (int j = y_offset ; j <= y_offset + 1; ++j){
              if (j < grid_height){
                 y_distance = 1 - ((scale_y_ * y + offset_y_ - j ) > 0.0 ? (scale_y_ * y + offset_y_- j) : -(scale_y_ * y + offset_y_ - j ));
                 
                for (int k = z_offset; k <= z_offset + 1; ++k){
                  if ( k < grid_channels ) {
                     Dtype z_abs = (depth_d_ * guidance_data[ guidance_offset] + offset_z_ - k );
                    z_distance =  1 - ( z_abs > 0.0 ? z_abs : -z_abs );

                    Dtype scale =  x_distance * y_distance * z_distance;
                    // LOG(ERROR) << z_distance << "," << x_distance << "," << y_distance;
                    for(int coeff_ind = 0; coeff_ind < coefficient_len_; ++coeff_ind){
                      // top_data[top[0] -> offset(n, coeff_ind, y, x)] += scale * grid_data[bottom[1] -> offset(n, k * coefficient_len_ + coeff_ind, j, i)];
                      top_data[((n * coefficient_len_ + coeff_ind) * img_height + y) * img_width + x] += scale * grid_data[int(((n * (coefficient_len_ * grid_channels) + (k * coefficient_len_ + coeff_ind)) * grid_height + j) * grid_width + i)];

                    }
                  }
                }
              }
            }
          }
        }
  //     }
  //   }
  // }
  }
}

template <typename Dtype>
void SlicingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype *guidance_diff = bottom[0] -> mutable_cpu_diff();
  Dtype *grid_diff = bottom[1] -> mutable_cpu_diff();
  const Dtype *grid_data = bottom[1] -> cpu_data();
  const Dtype *guidance_data = bottom[0] -> cpu_data();

  const Dtype *top_diff = top[0] -> cpu_diff();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int grid_height = bottom[1] -> height();
  const int grid_width = bottom[1] -> width();
  // const int N = bottom[0] -> num();
  caffe_set<Dtype>(bottom[0] -> count(), 0.0, guidance_diff);
  caffe_set<Dtype>(bottom[1] -> count(), 0.0, grid_diff);
  Dtype x_distance,y_distance, z_distance;
  int guidance_offset;
  // int counter = 0;
  const int count = bottom[0] -> count();
  const int grid_channels = bottom[1] -> channels() / coefficient_len_;
  for(int index = 0; index <count; ++index){
  // for (int n = 0; n < N; ++n){
  //   for(int x = 0; x < img_width; ++x){
  //     for(int y = 0; y < img_height; ++y){
    // guidance_offset = bottom[0] -> offset(n, 0, y, x);

    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);
            guidance_offset = index;    


             int x_offset = scale_x_ * x + offset_x_;
            int y_offset = scale_y_ * y + offset_y_;
            int z_offset = depth_d_ * guidance_data[ guidance_offset] + offset_z_;

            for (int i = x_offset ; i <= x_offset + 1; ++i){
              if (i < grid_width) {
                 x_distance = 1 - ((scale_x_ * x + offset_x_ - i ) > 0.0 ? (scale_x_ * x + offset_x_ - i ) : -(scale_x_ * x + offset_x_ - i ));
                 
                for (int j = y_offset ; j <= y_offset + 1; ++j){
                  if (j < grid_height){
                     y_distance = 1 - ((scale_y_ * y + offset_y_ - j ) > 0.0 ? (scale_y_ * y + offset_y_ - j) : -(scale_y_ * y + offset_y_ - j ));
                     
                    for (int k = z_offset; k <= z_offset + 1; ++k){
                      if ( k < grid_channels ) {
                        Dtype z_abs = (depth_d_ * guidance_data[ guidance_offset] + offset_z_ - k );
                        z_distance =  1 - ( z_abs > 0.0 ? z_abs : -z_abs );
                        
                        Dtype scale =  x_distance * y_distance * z_distance;
                        // if (counter < 100){
                        //   LOG(ERROR) << scale << "," << x_distance << "," << y_distance << "," << z_distance;
                        //   counter ++;
                        // }
                        
                        // w.r.t bilateral grid
                        for(int coeff_ind = 0; coeff_ind  < coefficient_len_; ++coeff_ind){
                          // grid_diff[bottom[1] -> offset(n, k * coefficient_len_ + coeff_ind, j, i)] +=  scale  * top_diff[top[0] -> offset(n, coeff_ind, y, x)];
                          // LOG(ERROR)<<((n * (coefficient_len_ * depth_d_) + (k * coefficient_len_ + coeff_ind)) * grid_height + j) * grid_width + i;
                          grid_diff[int(((n * (coefficient_len_ * grid_channels) + (k * coefficient_len_ + coeff_ind)) * grid_height + j) * grid_width + i)] +=  
                                              scale  * top_diff[((n * coefficient_len_ + coeff_ind) * img_height + y) * img_width + x];
                
                        }
                        
                        // w.r.t guidance map
                        Dtype scale2 = x_distance * y_distance ;
                        for(int coeff_ind = 0; coeff_ind  < coefficient_len_; ++coeff_ind){
                          Dtype tao_diff = z_abs > 0 ? -1 * depth_d_: depth_d_; 
                          guidance_diff[guidance_offset] += top_diff[top[0] -> offset(n, coeff_ind, y, x)]  
                                                                    * scale2 * grid_data[bottom[1] -> offset(n, k * coefficient_len_ + coeff_ind, j, i)] 
                                                                    * tao_diff;
                                                                    
                        }
                      }
                    }
                  }
                }
      //     }
      //   }
      //} 
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SlicingLayer);
#endif

INSTANTIATE_CLASS(SlicingLayer);
REGISTER_LAYER_CLASS(Slicing);

}  // namespace caffe
