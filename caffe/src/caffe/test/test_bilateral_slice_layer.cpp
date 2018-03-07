#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bilateral_slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class BilateralSlicingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BilateralSlicingLayerTest()
      : blob_bottom_guidance_(new Blob<Dtype>(2, 1, 3, 8)),
        blob_bottom_grid_(new Blob<Dtype>(2, 12, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    
    UniformFiller<Dtype> uniform_filler(filler_param);
    GaussianFiller<Dtype> gaussian_filler(filler_param);
    uniform_filler.Fill(this->blob_bottom_guidance_);
    gaussian_filler.Fill(this->blob_bottom_grid_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BilateralSlicingLayerTest() {
    delete blob_bottom_guidance_;
    delete blob_bottom_grid_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_guidance_;
  Blob<Dtype>* const blob_bottom_grid_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilateralSlicingLayerTest, TestDtypesAndDevices);

TYPED_TEST(BilateralSlicingLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_coefficient_len(3);
  shared_ptr<BilateralSlicingLayer<Dtype> > layer(
      new BilateralSlicingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 8);
  EXPECT_EQ(this->blob_top_->channels(), 3);
}

TYPED_TEST(BilateralSlicingLayerTest, TestSetUp2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_coefficient_len(4);
  shared_ptr<BilateralSlicingLayer<Dtype> > layer(
      new BilateralSlicingLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 8);
  EXPECT_EQ(this->blob_top_->channels(), 4);
}



TYPED_TEST(BilateralSlicingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  bool IS_VALID_CUDA = false;

#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
   LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_coefficient_len(3);

    shared_ptr<BilateralSlicingLayer<Dtype> > layer(
        new BilateralSlicingLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);


  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


TYPED_TEST(BilateralSlicingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_coefficient_len(4);
  BilateralSlicingLayer<Dtype>  layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701, 0.0, 0.02);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

    

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

// TYPED_TEST(BilateralSlicingLayerTest, TestGradientOffset) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_coefficient_len(4);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(2);
//   inner_product_param->set_offset_z(3);
//   BilateralSlicingLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_);

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

// TYPED_TEST(BilateralSlicingLayerTest, TestGradientScale) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_coefficient_len(4);
//   inner_product_param->set_scale_x(0.25);
//   inner_product_param->set_scale_y(0.25);
//   BilateralSlicingLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_);

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

// TYPED_TEST(BilateralSlicingLayerTest, TestGradientDepth) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_guidance_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_grid_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_coefficient_len(4);
//   inner_product_param->set_depth_d(2);
//   BilateralSlicingLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_);

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }



}  // namespace caffe
