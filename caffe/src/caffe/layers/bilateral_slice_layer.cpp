#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bilateral_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// bottom[0] is guidance map g, value in range of [0,1], in shape of [N, 1, H,W]
// bottom[1] is bilateral grid, in shape of [N, D * stride, H', w']

template <typename Dtype>
void BilateralSlicingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  coefficient_len_ = this->layer_param_.slicing_param().coefficient_len();
  scale_x_         = this->layer_param_.slicing_param().scale_x();
  scale_y_         = this->layer_param_.slicing_param().scale_y();
  depth_d_         = this->layer_param_.slicing_param().depth_d();
  offset_x_         = this->layer_param_.slicing_param().offset_x();
  offset_y_         = this->layer_param_.slicing_param().offset_y();
  offset_z_         = this->layer_param_.slicing_param().offset_z();
  CHECK(scale_x_ < 0 && scale_y_ < 0) << "Note scale_x or scale_y is not allowed";
  CHECK(offset_x_ == 0 && offset_x_ == 0 && offset_z_ == 0) << "Note offset_x or offset_y or offset_z is not allowed";
  CHECK(depth_d_ < 0 ) << "Note depth_d_ is not allowed";
  

  CHECK(bottom[1] -> channels() % coefficient_len_ == 0) << "bilateral grid channels must be divied by coefficient_len_";
  

  
}

template <typename Dtype>
void BilateralSlicingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0] -> Reshape(bottom[0] -> num(), coefficient_len_, bottom[0] -> height(), bottom[0] -> width());
}


template <typename Dtype>
void BilateralSlicingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void BilateralSlicingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
}

#ifdef CPU_ONLY
STUB_GPU(BilateralSlicingLayer);
#endif

INSTANTIATE_CLASS(BilateralSlicingLayer);
REGISTER_LAYER_CLASS(BilateralSlicing);

}  // namespace caffe
