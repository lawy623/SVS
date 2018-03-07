#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_division_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  epsilon_ = this->layer_param_.eltwise_division_param().epsilon();

}

template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0] -> count(), bottom[1] -> count()) << "bottom size must be equal";
  top[0] -> ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0] -> count();
  const Dtype *a_value = bottom[0] -> cpu_data();
  const Dtype *b_value = bottom[1] -> cpu_data();
  Dtype *top_value = top[0] -> mutable_cpu_data();
  for(int i = 0; i < count; ++i){
    if (std::abs(b_value[i]) < epsilon_){
      top_value[i] = 0;
    }else{
      top_value[i] = a_value[i] / b_value[i];
    }
  }
}

template <typename Dtype>
void EltwiseDivisionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0] -> count();
  const Dtype *a_value = bottom[0] -> cpu_data();
  const Dtype *b_value = bottom[1] -> cpu_data();
  const Dtype *top_diff = top[0] -> cpu_diff();
  Dtype *a_diff = bottom[0] -> mutable_cpu_diff();
  Dtype *b_diff = bottom[1] -> mutable_cpu_diff();

  for(int i = 0; i < count; ++i){
    if (std::abs(b_value[i]) < epsilon_){
      a_diff[i] = 0.0;
      b_diff[i] = 0.0;
    }else{
      a_diff[i] = top_diff[i] / b_value[i];
      // b_diff[i] = top_diff[i] * (-a_value[i]) / (b_value[i]*b_value[i]);
      b_diff[i] = top_diff[i] * (-a_value[i]) * std::pow(b_value[i], -2);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseDivisionLayer);
#endif

INSTANTIATE_CLASS(EltwiseDivisionLayer);
REGISTER_LAYER_CLASS(EltwiseDivision);

}  // namespace caffe
