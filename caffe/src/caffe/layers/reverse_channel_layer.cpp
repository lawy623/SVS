#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/reverse_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReverseChannelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

INSTANTIATE_CLASS(ReverseChannelLayer);
REGISTER_LAYER_CLASS(ReverseChannel);

}  // namespace caffe
