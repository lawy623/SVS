#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/grid_splat_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class GridSplatLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  GridSplatLayerTest()
      : blob_bottom_img_(new Blob<Dtype>(2, 1, 4, 6)),
        blob_bottom_edge_(new Blob<Dtype>(2, 1, 4, 6)),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_weight_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_max(0.9);
    UniformFiller<Dtype> uniform_filler(filler_param);
    GaussianFiller<Dtype> gaussian_filler(filler_param);
    uniform_filler.Fill(this->blob_bottom_img_);
    uniform_filler.Fill(this->blob_bottom_edge_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_weight_);
  }
  virtual ~GridSplatLayerTest() {
    delete blob_bottom_img_;
    delete blob_bottom_edge_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_img_;
  Blob<Dtype>* const blob_bottom_edge_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_weight_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GridSplatLayerTest, TestDtypesAndDevices);


TYPED_TEST(GridSplatLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  LayerParameter layer_param;
  // SlicingPatameter* inner_product_param =
  //     layer_param.mutable_slicing_param();
  shared_ptr<GridSplatLayer<Dtype> > layer(
      new GridSplatLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 2);
  EXPECT_EQ(this->blob_top_data_->height(), 6);
  EXPECT_EQ(this->blob_top_data_->width(), 8);
  EXPECT_EQ(this->blob_top_data_->channels(), 2);
}

TYPED_TEST(GridSplatLayerTest, TestSetUp2) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> shape = this->blob_bottom_img_ -> shape();
  shape[1] = 3;
  this->blob_bottom_img_ -> Reshape(shape);

  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_scale_x(0.5);
  inner_product_param->set_scale_y(0.5);
  inner_product_param->set_depth_d(3);
  inner_product_param->set_offset_x(3);
  inner_product_param->set_offset_y(3);
  inner_product_param->set_offset_z(3);
  
  GridSplatLayerParameter* grid_splat_param =
      layer_param.mutable_grid_splat_param();
  grid_splat_param -> set_edge_min(0.0);
  grid_splat_param -> set_edge_max(1.0);


  shared_ptr<GridSplatLayer<Dtype> > layer(
      new GridSplatLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->shape(0), 2);
  EXPECT_EQ(this->blob_top_data_->shape(1), 3);
  EXPECT_EQ(this->blob_top_data_->shape(2), 3 + 7);
  EXPECT_EQ(this->blob_top_data_->shape(3), 2 + 7);
  EXPECT_EQ(this->blob_top_data_->shape(4), 3 + 7);
}

TYPED_TEST(GridSplatLayerTest, TestGradientMultiChannelsSoft) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> shape = this->blob_bottom_img_ -> shape();
  shape[1] = 2;
  this->blob_bottom_img_ -> Reshape(shape);
  FillerParameter filler_param;
   UniformFiller<Dtype> uniform_filler(filler_param);
    // GaussianFiller<Dtype> gaussian_filler(filler_param);
   uniform_filler.Fill(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_scale_x(0.5);
  inner_product_param->set_scale_y(0.5);
  inner_product_param->set_depth_d(3);
  inner_product_param->set_offset_x(3);
  inner_product_param->set_offset_y(3);
  inner_product_param->set_offset_z(3);
  
  GridSplatLayerParameter* grid_splat_param =
      layer_param.mutable_grid_splat_param();
  grid_splat_param -> set_edge_min(0.0);
  grid_splat_param -> set_edge_max(1.0);
  grid_splat_param -> set_weight_mode(GridSplatLayerParameter_WeightMode_SOFT_MODE);


  GridSplatLayer<Dtype>  layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

    

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


TYPED_TEST(GridSplatLayerTest, TestGradientMultiChannelsSoftOneInput) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> shape = this->blob_bottom_img_ -> shape();
  shape[1] = 2;
  this->blob_bottom_img_ -> Reshape(shape);
  FillerParameter filler_param;
   UniformFiller<Dtype> uniform_filler(filler_param);
    // GaussianFiller<Dtype> gaussian_filler(filler_param);
   uniform_filler.Fill(this->blob_bottom_img_);
   
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
  // this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_scale_x(0.5);
  inner_product_param->set_scale_y(0.5);
  inner_product_param->set_depth_d(3);
  inner_product_param->set_offset_x(3);
  inner_product_param->set_offset_y(3);
  inner_product_param->set_offset_z(3);
  
  GridSplatLayerParameter* grid_splat_param =
      layer_param.mutable_grid_splat_param();
  grid_splat_param -> set_edge_min(0.0);
  grid_splat_param -> set_edge_max(1.0);
  grid_splat_param -> set_weight_mode(GridSplatLayerParameter_WeightMode_SOFT_MODE);


  GridSplatLayer<Dtype>  layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

    

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

// passed 
// TYPED_TEST(GridSplatLayerTest, TestGradientMuliChannels) {
//   typedef typename TypeParam::Dtype Dtype;
//   vector<int> shape = this->blob_bottom_img_ -> shape();
//   shape[1] = 2;
//   this->blob_bottom_img_ -> Reshape(shape);
//   FillerParameter filler_param;
//    UniformFiller<Dtype> uniform_filler(filler_param);
//     // GaussianFiller<Dtype> gaussian_filler(filler_param);
//    uniform_filler.Fill(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);


//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 0);

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

TYPED_TEST(GridSplatLayerTest, TestGradientMuliChannelsPerEdge) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.clear();

  vector<int> shape = this->blob_bottom_img_ -> shape();
  shape[1] = 2;
  this->blob_bottom_img_ -> Reshape(shape);
  FillerParameter filler_param;
   UniformFiller<Dtype> uniform_filler(filler_param);
    // GaussianFiller<Dtype> gaussian_filler(filler_param);
   uniform_filler.Fill(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);

  shape = this->blob_bottom_edge_ -> shape();
  shape[1] = 2;
  this->blob_bottom_edge_ -> Reshape(shape);
  uniform_filler.Fill(this->blob_bottom_edge_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_scale_x(0.5);
  inner_product_param->set_scale_y(0.5);
  inner_product_param->set_depth_d(3);
  inner_product_param->set_offset_x(3);
  inner_product_param->set_offset_y(3);
  inner_product_param->set_offset_z(3);
  
  GridSplatLayerParameter* grid_splat_param =
      layer_param.mutable_grid_splat_param();
  grid_splat_param -> set_edge_min(0.0);
  grid_splat_param -> set_edge_max(1.0);
  grid_splat_param -> set_weight_mode(GridSplatLayerParameter_WeightMode_SOFT_MODE);

  GridSplatLayer<Dtype>  layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

    

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


// TYPED_TEST(GridSplatLayerTest, TestForward) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
//   bool IS_VALID_CUDA = false;

// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//    LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);

//     shared_ptr<GridSplatLayer<Dtype> > layer(
//         new GridSplatLayer<Dtype>(layer_param));
//     layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);


//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }




// TYPED_TEST(GridSplatLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);


//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 0);

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }
// passed
TYPED_TEST(GridSplatLayerTest, TestGradientSoft) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
  LayerParameter layer_param;
  SlicingPatameter* inner_product_param =
      layer_param.mutable_slicing_param();
  inner_product_param->set_scale_x(0.5);
  inner_product_param->set_scale_y(0.5);
  inner_product_param->set_depth_d(3);
  inner_product_param->set_offset_x(3);
  inner_product_param->set_offset_y(3);
  inner_product_param->set_offset_z(3);
  
  GridSplatLayerParameter* grid_splat_param =
      layer_param.mutable_grid_splat_param();
  grid_splat_param -> set_edge_min(0.0);
  grid_splat_param -> set_edge_max(1.0);
  grid_splat_param -> set_weight_mode(GridSplatLayerParameter_WeightMode_SOFT_MODE);


  GridSplatLayer<Dtype>  layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

    

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

// TYPED_TEST(GridSplatLayerTest, TestGradientSoftEdge) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);
//   grid_splat_param -> set_weight_mode(GridSplatLayerParameter_WeightMode_SOFT_MODE);


//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 1);

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }


// TYPED_TEST(GridSplatLayerTest, TestGradientOneInput) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   // this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
//   this -> blob_top_vec_.clear();
//   this -> blob_top_vec_.push_back(this->blob_top_data_);
//   this -> blob_top_vec_.push_back(this->blob_top_weight_);

//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);


//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_,0);
//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }


// TYPED_TEST(GridSplatLayerTest, TestGradientTwoOutput) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
//   this -> blob_top_vec_.clear();
//   this -> blob_top_vec_.push_back(this->blob_top_data_);
//   this -> blob_top_vec_.push_back(this->blob_top_weight_);
  
//   bool IS_VALID_CUDA = false;
// #ifndef CPU_ONLY
//   IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
// #endif
//   if (Caffe::mode() == Caffe::CPU ||
//       sizeof(Dtype) == 4 || IS_VALID_CUDA) {
//   LayerParameter layer_param;
//   SlicingPatameter* inner_product_param =
//       layer_param.mutable_slicing_param();
//   inner_product_param->set_scale_x(0.5);
//   inner_product_param->set_scale_y(0.5);
//   inner_product_param->set_depth_d(3);
//   inner_product_param->set_offset_x(3);
//   inner_product_param->set_offset_y(3);
//   inner_product_param->set_offset_z(3);
  
//   GridSplatLayerParameter* grid_splat_param =
//       layer_param.mutable_grid_splat_param();
//   grid_splat_param -> set_edge_min(0.0);
//   grid_splat_param -> set_edge_max(1.0);


//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_,0);
//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

// TYPED_TEST(GridSplatLayerTest, TestGradientOffset) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
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
//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 0 );

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

// TYPED_TEST(GridSplatLayerTest, TestGradientOffset2) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
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
//   inner_product_param->set_offset_x(0);
//   inner_product_param->set_offset_y(0);
//   inner_product_param->set_offset_z(3);
//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_, 0 );

    

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }



// TYPED_TEST(GridSplatLayerTest, TestGradientScale) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
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
//   inner_product_param->set_scale_x(0.3);
//   inner_product_param->set_scale_y(0.25);
//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_);

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }

// TYPED_TEST(GridSplatLayerTest, TestGradientDepth) {
//   typedef typename TypeParam::Dtype Dtype;
//   this->blob_bottom_vec_.push_back(this->blob_bottom_img_);
//   this->blob_bottom_vec_.push_back(this->blob_bottom_edge_);
  
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
//   GridSplatLayer<Dtype>  layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//     GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0.0, 0.02);
//     checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//         this->blob_top_vec_);

//   } else {
//     LOG(ERROR) << "Skipping test due to old architecture.";
//   }
// }



}  // namespace caffe
