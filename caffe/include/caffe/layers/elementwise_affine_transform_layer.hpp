#ifndef CAFFE_EltwiseAffineTransformLayer_
#define CAFFE_EltwiseAffineTransformLayer_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <algorithm>
#include <math.h>
#include <stdlib.h>

namespace caffe {

/**
 * @brief  Slicing layer for Deep Bilateral Learning for Real-Time Image Enhancement
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseAffineTransformLayer : public Layer<Dtype> {
 public:
  explicit EltwiseAffineTransformLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EltwiseAffineTransform"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }  // bottom[0] is coefficient map which are in shape of [N, 12, H,W]
                                                                // bottom[1] is input image, which are in shape of [N,3, H,W]
                                                                // NOTE that x is along Width, y is along height.
  virtual inline int ExactNumTopBlobs() const { return 1; }     // top[0] is the transformed output, in shape of [N, 3, H,W]

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // Note the function does not have differential value when scale = 0
  inline Dtype tao_f(Dtype scale) const {return std::max(0.0, 1 - fabs(scale)); };
  int n_theta_;
  int n_out_channels_;
};

}  // namespace caffe

#endif  // CAFFE_EltwiseAffineTransformLayer_
