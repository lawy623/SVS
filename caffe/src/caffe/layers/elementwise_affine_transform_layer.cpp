#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/elementwise_affine_transform_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// bottom[0] is coefficient map which are in shape of [N, 12, H,W]
// bottom[1] is input image, which are in shape of [N,3, H,W]
template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[0]->num(), bottom[1] -> num());
  CHECK_EQ(bottom[0]->channels(), 12);
  CHECK_EQ(bottom[1]->channels(), 3);
  CHECK_EQ(bottom[0]->height(), bottom[1] -> height());
  CHECK_EQ(bottom[0]->width(), bottom[1] -> width());
  n_out_channels_ = 3;
  n_theta_ = 3;
  
}

template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0] -> ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype *coeff_data = bottom[0] -> cpu_data();
  const Dtype *img_data = bottom[1] -> cpu_data();
  Dtype *top_data = top[0] -> mutable_cpu_data();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
  const int count = bottom[0] -> count() / (bottom[0] -> channels());
  int n_out_channels = n_out_channels_;
  int n_theta = n_theta_;
  caffe_set<Dtype>(top[0] -> count(), 0.0, top_data);
  for(int index = 0; index < count; ++index){
     // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);

    for(int c = 0; c < n_out_channels; c++ ){
      int out_offset = ((n * n_out_channels + c) * img_height + y) * img_width + x;
      int T_offset = ((n * (n_theta  + 1) * n_out_channels + n_theta + (n_theta + 1) * c) * img_height + y) * img_width + x;
      top_data[out_offset] = coeff_data[T_offset];
      int img_offset = ((n * n_theta + 0) * img_height + y) * img_width + x;
      int R_offset = ((n * (n_theta  + 1) * n_out_channels + 0 + (n_theta + 1) * c) * img_height + y) * img_width + x;
      for (int c_head = 0; c_head <= n_theta - 1; ++c_head){
        top_data[out_offset] += img_data[img_offset] * coeff_data[R_offset];
        img_offset += img_height * img_width;
        R_offset += img_width * img_height;
      }
      // LOG(ERROR) << top_data[out_offset] << "," << out_offset;
    }
  }
}

template <typename Dtype>
void EltwiseAffineTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype *coeff_diff = bottom[0] -> mutable_cpu_diff();
  Dtype *img_diff = bottom[1] -> mutable_cpu_diff();
  const Dtype *coeff_data = bottom[0] -> cpu_data();
  const Dtype *img_data = bottom[1] -> cpu_data();

  const Dtype *top_diff = top[0] -> cpu_diff();
  const int img_height = bottom[0] -> height();
  const int img_width =  bottom[0] -> width();
 
  caffe_set<Dtype>(bottom[0] -> count(), 0.0, coeff_diff);
  caffe_set<Dtype>(bottom[1] -> count(), 0.0, img_diff);
  int n_out_channels = n_out_channels_;
  int n_theta = n_theta_;

  const int count = bottom[0] -> count() / (bottom[0] -> channels());
  for(int index = 0; index < count; ++index){
     // ((n * channels() + c) * height() + h) * width() + w
    int x = index % img_width;
    int y = (index / img_width) % img_height;
    int n = index / (img_width * img_height);

    for(int c = 0; c < n_out_channels; c++ ){
      int out_offset = ((n * n_out_channels + c) * img_height + y) * img_width + x;
      int T_offset = ((n * (n_theta  + 1) * n_out_channels + n_theta + (n_theta + 1) * c) * img_height + y) * img_width + x;
      coeff_diff[T_offset] = top_diff[out_offset];
      int img_offset = ((n * n_theta + 0) * img_height + y) * img_width + x;
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


#ifdef CPU_ONLY
STUB_GPU(EltwiseAffineTransformLayer);
#endif

INSTANTIATE_CLASS(EltwiseAffineTransformLayer);
REGISTER_LAYER_CLASS(EltwiseAffineTransform);

}  // namespace caffe
