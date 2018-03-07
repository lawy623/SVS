// Copyright 2015 Zhu.Jin Liang

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/util_pre_define.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
Mat_<Dtype> Get_Affine_matrix(
		const Point_<Dtype>& srcCenter, 
		const Point_<Dtype>& dstCenter,
    const Dtype alpha, 
    const Dtype scale) 
{
  Mat_<Dtype> M(2, 3);

  M(0, 0) = scale * cos(alpha);
  M(0, 1) = scale * sin(alpha);
  M(1, 0) = -M(0, 1);
  M(1, 1) = M(0, 0);

  M(0, 2) = srcCenter.x - M(0, 0) 
  		* dstCenter.x - M(0, 1) * dstCenter.y;
  M(1, 2) = srcCenter.y - M(1, 0) 
  		* dstCenter.x - M(1, 1) * dstCenter.y;
  return M;
}
template Mat_<float> Get_Affine_matrix(
		const Point_<float>& srcCenter, 
		const Point_<float>& dstCenter,
    const float alpha, 
    const float scale);
template Mat_<double> Get_Affine_matrix(
		const Point_<double>& srcCenter, 
		const Point_<double>& dstCenter,
    const double alpha, 
    const double scale);

template <typename Dtype>
Mat_<Dtype> inverseMatrix(const Mat_<Dtype>& M) {
	Dtype D = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  D = D != 0 ? 1. / D : 0;

  Mat_<Dtype> inv_M(2, 3);

  inv_M(0, 0) = M(1, 1) * D;
  inv_M(0, 1) = M(0, 1) * (-D);
  inv_M(1, 0) = M(1, 0) * (-D);
  inv_M(1, 1) = M(0, 0) * D;

  inv_M(0, 2) = -inv_M(0, 0) * M(0, 2) 
  		- inv_M(0, 1) * M(1, 2);
  inv_M(1, 2) = -inv_M(1, 0) * M(0, 2) 
  		- inv_M(1, 1) * M(1, 2);
  return inv_M;
}
template Mat_<float> inverseMatrix(const Mat_<float>& M);
template Mat_<double> inverseMatrix(const Mat_<double>& M);

template <typename Dtype>
void mAffineWarp(const Mat_<Dtype>& M, 
		const Mat& srcImg, 
		Mat& dstImg, 
		const bool fill_type,
    const uchar value) 
{
  if (dstImg.empty()) {
  	dstImg = cv::Mat(srcImg.size(), 
  			srcImg.type(), cv::Scalar::all(0));
  }

  for (int y = 0; y < dstImg.rows; ++y) {
    for (int x = 0; x < dstImg.cols; ++x) {
      Dtype fx = M(0, 0) * x + M(0, 1) * y + M(0, 2);
      Dtype fy = M(1, 0) * x + M(1, 1) * y + M(1, 2);
      int sy = cvFloor(fy);
      int sx = cvFloor(fx);
      if (fill_type && (sy < 1 || sy > srcImg.rows - 2 
      		|| sx < 1 || sx > srcImg.cols - 2)) 
      {
        for (int k = 0; k < srcImg.channels(); ++k) {
          dstImg.at<cv::Vec3b>(y, x)[k] = value;
        }
        continue;
      }

      fx -= sx;
      fy -= sy;

      sy = MAX(1, MIN(sy, srcImg.rows - 2)); //my modify
      sx = MAX(1, MIN(sx, srcImg.cols - 2)); //my modify
      Dtype w_y0 = std::abs(1.0f - fy);
      Dtype w_y1 = std::abs(fy);
      Dtype w_x0 = std::abs(1.0f - fx);
      Dtype w_x1 = std::abs(fx);
      for (int k = 0; k < srcImg.channels(); ++k) {
        dstImg.at<cv::Vec3b>(y, x)[k] = 
        	(srcImg.at<cv::Vec3b>(sy, sx)[k] * w_x0 * w_y0
            + srcImg.at<cv::Vec3b>(sy + 1, sx)[k] * w_x0 * w_y1
            + srcImg.at<cv::Vec3b>(sy, sx + 1)[k] * w_x1 * w_y0
            + srcImg.at<cv::Vec3b>(sy + 1, sx + 1)[k] * w_x1 * w_y1);
      }
    }
  }
}
template void mAffineWarp<float>(
		const Mat_<float>& M, 
		const Mat& srcImg, 
		Mat& dstImg, 
		const bool fill_type,
    const uchar value);
template void mAffineWarp<double>(
		const Mat_<double>& M, 
		const Mat& srcImg, 
		Mat& dstImg, 
		const bool fill_type,
    const uchar value);


template <typename Dtype>
void BiLinearResizeMat_cpu(
		const Dtype* src, 
		const int src_height, 
		const int src_width,
		Dtype* dst, 
		const int dst_height, 
		const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	int loop_n = dst_height * dst_width;
	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;

		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name() )
		{
			src_h = floor(fh);
		}
		else
		{
			src_h = floorf(fh);
		}

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		int src_w ;//= floor(fw);
		if(typeid(Dtype).name() == typeid(double).name() )
		{
			src_w = floor(fw);
		}
		else
		{
			src_w = floorf(fw);
		}
		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		dst_data[dst_idx] += (w_h0 * w_w0 * src_data[src_idx]);

		if (src_w + 1 < src_width) {
			dst_data[dst_idx] += 
					(w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height) {
			dst_data[dst_idx] += 
					(w_h1 * w_w0 * src_data[src_idx + src_width]);
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height) {
			dst_data[dst_idx] += 
					(w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}
	}
}
template void BiLinearResizeMat_cpu(
		const float* src, 
		const int src_height, 
		const int src_width,
		float* dst, 
		const int dst_height, 
		const int dst_width);
template void BiLinearResizeMat_cpu(
		const double* src, 
		const int src_height, 
		const int src_width,
		double* dst, 
		const int dst_height, 
		const int dst_width);

template <typename Dtype>
void RuleBiLinearResizeMat_cpu(
		const Dtype* src,
		Dtype* dst, 
		const int dst_h, 
		const int dst_w,
		const Dtype* loc1, 
		const Dtype* weight1, 
		const Dtype* loc2,
		const Dtype* weight2,
		const	Dtype* loc3,
		const Dtype* weight3,
		const Dtype* loc4, 
		const Dtype* weight4)
{

	Dtype* dst_data = dst;
	const Dtype* src_data = src;

	int loop_n = dst_h  * dst_w ;
	for(int i=0 ; i< loop_n; i++)
	{
		dst_data[i] += 
				(weight1[i] * src_data[static_cast<int>(loc1[i])]);
		dst_data[i] += 
				(weight2[i] * src_data[static_cast<int>(loc2[i])]);
		dst_data[i] += 
				(weight3[i] * src_data[static_cast<int>(loc3[i])]);
		dst_data[i] += 
				(weight4[i] * src_data[static_cast<int>(loc4[i])]);
	}
}
template void RuleBiLinearResizeMat_cpu(
		const float* src,
		float* dst, 
		const int dst_h, 
		const int dst_w,
		const float* loc1, 
		const float* weight1, 
		const float* loc2,
		const float* weight2,
		const	float* loc3,
		const float* weight3,
		const float* loc4,
		const float* weight4);
template void RuleBiLinearResizeMat_cpu(
		const double* src,
		double* dst, 
		const int dst_h, 
		const int dst_w,
		const double* loc1, 
		const double* weight1, 
		const double* loc2,
		const double* weight2,
		const	double* loc3,
		const double* weight3,
		const double* loc4, 
		const double* weight4);

template <typename Dtype>
void GetBiLinearResizeMatRules_cpu( 
	  const int src_height, 
	  const int src_width,
		const int dst_height, 
		const int dst_width,
		Dtype* loc1, Dtype* weight1, 
		Dtype* loc2, Dtype* weight2,
		Dtype* loc3, Dtype* weight3, 
		Dtype* loc4, Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;

	int loop_n = dst_height * dst_width;
	caffe::caffe_set(loop_n,(Dtype)0,loc1);
	caffe::caffe_set(loop_n,(Dtype)0,loc2);
	caffe::caffe_set(loop_n,(Dtype)0,loc4);
	caffe::caffe_set(loop_n,(Dtype)0,loc3);

	caffe::caffe_set(loop_n,(Dtype)0,weight1);
	caffe::caffe_set(loop_n,(Dtype)0,weight2);
	caffe::caffe_set(loop_n,(Dtype)0,weight3);
	caffe::caffe_set(loop_n,(Dtype)0,weight4);

	for(int i=0 ; i< loop_n; i++)
	{
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		int src_h ;
		if(typeid(Dtype).name() == typeid(double).name())
			 src_h = floor(fh);
		else
			 src_h = floorf(fh);

		fh -= src_h;
		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;

		int src_w ;
		if(typeid(Dtype).name() == typeid(double).name())
			src_w = floor(fw);
		else
			src_w = floorf(fw);

		fw -= src_w;
		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
		// dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = static_cast<Dtype>(src_idx);
		weight1[dst_idx] = w_h0 * w_w0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = static_cast<Dtype>(src_idx + 1);
			weight2[dst_idx] = w_h0 * w_w1;
			// dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
			// dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = static_cast<Dtype>(src_idx + src_width);
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = static_cast<Dtype>(src_idx + src_width + 1);
			weight4[dst_idx] = w_h1 * w_w1;
			// dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}
	}
}
template void GetBiLinearResizeMatRules_cpu(  
		const int src_height, 
		const int src_width,
		 const int dst_height, 
		 const int dst_width,
		float* loc1, 
		float* weight1, 
		float* loc2, 
		float* weight2,
		float* loc3, 
		float* weight3, 
		float* loc4, 
		float* weight4);
template void GetBiLinearResizeMatRules_cpu(  
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width,
		double* loc1, 
		double* weight1,
		double* loc2, 
		double* weight2,
		double* loc3, 
		double* weight3, 
		double* loc4, 
		double* weight4);

template <typename Dtype>
void ResizeBlob_cpu(
		const Blob<Dtype>* src, 
		const int src_n, 
		const int src_c,
		Blob<Dtype>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff /*true: data, false: diff */) 
{
	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = 
			(src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = 
			(dst_n * dst_channels + dst_c) * dst_height * dst_width;


	const Dtype* src_data = 
			data_or_diff ? 
			&(src->cpu_data()[src_offset]) :
			&(src->cpu_diff()[src_offset]);
	Dtype* dst_data = 
			data_or_diff ? 
			&(dst->mutable_cpu_data()[dst_offset]) :
			&(dst->mutable_cpu_diff()[dst_offset]);

	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
}
template void ResizeBlob_cpu(
		const Blob<float>* src, 
		const int src_n, 
		const int src_c,
		Blob<float>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff);
template void ResizeBlob_cpu(
		const Blob<double>* src, 
		const int src_n, 
		const int src_c,
		Blob<double>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff);

template <typename Dtype>
void ResizeBlob_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())
			<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())
			<< "src->channels() == dst->channels()";

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src, n , c , dst, n, c);
		}
	}
}
template void ResizeBlob_cpu(
		const Blob<float>* src, Blob<float>* dst);
template void ResizeBlob_cpu(
		const Blob<double>* src, Blob<double>* dst);

template <typename Dtype>
void ResizeBlob_Data_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())
			<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())
			<< "src->channels() == dst->channels()";

	const bool data_or_diff = true;
	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c, dst,n,c, data_or_diff);
		}
	}
}
template void ResizeBlob_Data_cpu(
		const Blob<float>* src, Blob<float>* dst);
template void ResizeBlob_Data_cpu(
		const Blob<double>* src, Blob<double>* dst);

template <typename Dtype>
void ResizeBlob_Diff_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst)
{
	CHECK(src->num() == dst->num())
			<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())
			<< "src->channels() == dst->channels()";

	const bool data_or_diff = false;
	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c, dst,n,c, data_or_diff);
		}
	}
}
template void ResizeBlob_Diff_cpu(
		const Blob<float>* src, Blob<float>* dst);
template void ResizeBlob_Diff_cpu(
		const Blob<double>* src, Blob<double>* dst);

template <typename Dtype>
void ResizeBlob_cpu(
		const Blob<Dtype>* src,
		Blob<Dtype>* dst,
		Blob<Dtype>* loc1, 
		Blob<Dtype>* loc2, 
		Blob<Dtype>* loc3, 
		Blob<Dtype>* loc4)
{
	CHECK(src->num() == dst->num())
			<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())
			<< "src->channels() == dst->channels()";

	GetBiLinearResizeMatRules_cpu(
			src->height(),
			src->width(),
			dst->height(), 
			dst->width(),
			loc1->mutable_cpu_data(), 
			loc1->mutable_cpu_diff(), 
			loc2->mutable_cpu_data(), 
			loc2->mutable_cpu_diff(),
			loc3->mutable_cpu_data(), 
			loc3->mutable_cpu_diff(), 
			loc4->mutable_cpu_data(), 
			loc4->mutable_cpu_diff());

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_cpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_cpu(
		const Blob<float>* src,
		Blob<float>* dst,
		Blob<float>* loc1, 
		Blob<float>* loc2, 
		Blob<float>* loc3, 
		Blob<float>* loc4);
template void ResizeBlob_cpu(
		const Blob<double>* src,
		Blob<double>* dst,
		Blob<double>* loc1, 
		Blob<double>* loc2, 
		Blob<double>* loc3, 
		Blob<double>* loc4);

void CropAndResizePatch(
		const cv::Mat& src, 
		cv::Mat& dst, 
		const vector<float>& coords,
		const cv::Size& resize_size, 
		const bool is_fill, 
		const int fill_value) 
{
	CHECK_EQ(src.channels(), 3);
  const int x1 = MIN(src.cols - 1, MAX(0, coords[0]));
  const int y1 = MIN(src.rows - 1, MAX(0, coords[1]));
  const int x2 = MIN(src.cols - 1, MAX(0, coords[2]));
  const int y2 = MIN(src.rows - 1, MAX(0, coords[3]));

	cv::Mat dst_tmp;
	if (is_fill) {
		const int crop_width = coords[2] - coords[0] + 1;
		const int crop_height = coords[3] - coords[1] + 1;
		const size_t nCols = sizeof(uchar) * MIN(x2 - x1 + 1, crop_width) * 3;
		int dst_tmp_h_beg = 0, dst_tmp_w_beg = 0;
		if (coords[0] < 0) dst_tmp_w_beg = -(int)coords[0];
		if (coords[1] < 0) dst_tmp_h_beg = -(int)coords[1];

		dst_tmp = cv::Mat(crop_height, crop_width, CV_8UC3,
				cv::Scalar(fill_value, fill_value, fill_value));

		for (int crop_y1 = y1, h = dst_tmp_h_beg; 
				h < crop_height && crop_y1 <= y2; ++crop_y1, ++h) 
		{
			memcpy(
					dst_tmp.ptr<uchar>(h) + 3 * dst_tmp_w_beg, 
					src.ptr<uchar>(crop_y1) + 3 * x1,
					nCols);
		}
	} else {
		cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
		dst_tmp = src(roi);
	}

	cv::resize(dst_tmp, dst, resize_size, 0, 0, cv::INTER_LINEAR);
}

// template <typename Dtype>
// void ImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image) {
//   const int height = blob->height();
//   const int width = blob->width();
//   const int channels = blob->channels();
//   CHECK_EQ(channels, 3);
//   const int dims = channels * height * width;
//   const int size = height * width;

//   Dtype* data = blob->mutable_cpu_data();
// 	for (int h = 0, offset_h = n * dims; h < height; ++h, offset_h += width) {
// 		const uchar* img_ptr = image.ptr<uchar>(h);
// 		for (int w = 0, offset_w = offset_h; w < width; ++w, ++offset_w) {
// 			for (int c = 0, top_index = offset_w; c < channels; ++c, top_index += size) {
//         data[top_index] = static_cast<Dtype>(*img_ptr++);
//       }
//     }
//   }
// }
template <typename Dtype>
void ImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image) {
  const int height = blob->height();
  const int width = blob->width();
  const int channels = blob->channels();
  const int img_width = image.cols;
  const int img_height = image.rows;
  const int img_channels = image.channels();
  // CHECK_EQ(channels, 3);
  CHECK_LE(img_width, width);
  CHECK_LE(img_height, height);
  CHECK_EQ(channels, img_channels);

  const int size = height * width;
  const int dims = channels * height * width;
  Dtype* data = blob->mutable_cpu_data();
	for (int h = 0, offset_h = n * dims; h < img_height; ++h, offset_h += width) {
		const uchar* img_ptr = image.ptr<uchar>(h);
		for (int w = 0, offset_w = offset_h; w < img_width; ++w, ++offset_w) {
			for (int c = 0, top_index = offset_w; c < channels; ++c, top_index += size) {
        data[top_index] = static_cast<Dtype>(*img_ptr++);
      }
    }
  }
}
template void ImageDataToBlob(Blob<float>* blob, const int n, const cv::Mat& image);
template void ImageDataToBlob(Blob<double>* blob, const int n, const cv::Mat& image);

template <typename Dtype>
void ImageDataToBlob(Blob<Dtype>* blob, const int n, const cv::Mat& image,
		const std::vector<Dtype> mean_values) 
{
  const int height = blob->height();
  const int width = blob->width();
  const int channels = blob->channels();
  const int img_width = image.cols;
  const int img_height = image.rows;
  const int img_channels = image.channels();
  CHECK_EQ(channels, 3);
  CHECK_LE(img_width, width);
  CHECK_LE(img_height, height);
  CHECK_EQ(channels, img_channels);

  const int size = height * width;
  const int dims = channels * height * width;
  Dtype* data = blob->mutable_cpu_data();
  CHECK(mean_values.size() == 3);
	for (int h = 0, offset_h = n * dims; h < img_height; ++h, offset_h += width) {
		const uchar* img_ptr = image.ptr<uchar>(h);
		for (int w = 0, offset_w = offset_h; w < img_width; ++w, ++offset_w) {
			for (int c = 0, top_index = offset_w; c < channels; ++c, top_index += size) {
        data[top_index] = static_cast<Dtype>(*img_ptr++) - mean_values[c];
      }
    }
  }
}
template void ImageDataToBlob(Blob<float>* blob, const int n, const cv::Mat& image,
		const std::vector<float> mean_values);
template void ImageDataToBlob(Blob<double>* blob, const int n, const cv::Mat& image,
		const std::vector<double> mean_values);

template <typename Dtype>
cv::Mat BlobToColorImage(const Blob<Dtype>* blob, const int n) {
	CHECK_EQ(blob->channels(), 3) << "Only Support Color images";
	cv::Mat img(blob->height(), blob->width(), CV_8UC3);
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < img.rows; ++h) {
			for (int w = 0; w < img.cols; ++w) {
				Dtype v1 = blob->data_at(n, c, h, w);
				uchar v2 = 0;
				if (v1 < 0) v2 = 0;
				else if (v1 > 255) v2 = 255;
				else v2 = static_cast<uchar>(v1);
				img.at<cv::Vec3b>(h, w)[c] = v2;
			}
		}
	}

	return img;
}
template cv::Mat BlobToColorImage(const Blob<float>* blob, const int n);
template cv::Mat BlobToColorImage(const Blob<double>* blob, const int n);

template <typename Dtype>
cv::Mat BlobToGrayImage(const Blob<Dtype>* blob, 
			const int n, const int c,
		const Dtype scale = Dtype(1.0)) {

	CHECK_GE(n, 0)  << "invalid n";
	CHECK_GE(c, 0)  << "invalid c";
	
	cv::Mat img(blob->height(), blob->width(), CV_8UC1);
	
	for (int h = 0; h < img.rows; ++h) {
		for (int w = 0; w < img.cols; ++w) {
			Dtype v1 = blob->data_at(n, c, h, w);
			v1 *= scale;
			uchar v2 = 0;
			if (v1 < 0) v2 = 0;
			else if (v1 > 255) v2 = 255;
			
			else v2 = static_cast<uchar>(v1);	
			// if((int)v2 > 0)
			// 	LOG(INFO) << (int)v2;

			img.at<uchar>(h, w) = v2;
		}
	}
	
	return img;
}
template cv::Mat BlobToGrayImage(const Blob<float>* blob, const int n,const int c,const float scale);
template cv::Mat BlobToGrayImage(const Blob<double>* blob, const int n,const int c,const double scale);


template <typename Dtype>
cv::Mat BlobToColorImage(const Blob<Dtype>* blob, const int n,
		const std::vector<Dtype> mean_values) 
{
	CHECK_EQ(blob->channels(), 3) << "Only Support Color images";
	cv::Mat img(blob->height(), blob->width(), CV_8UC3);
	CHECK(mean_values.size() == 3);
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < img.rows; ++h) {
			for (int w = 0; w < img.cols; ++w) {
				Dtype v1 = blob->data_at(n, c, h, w) + mean_values[c];
				uchar v2 = 0;
				if (v1 < 0) v2 = 0;
				else if (v1 > 255) v2 = 255;
				else v2 = static_cast<uchar>(v1);
				img.at<cv::Vec3b>(h, w)[c] = v2;
				// + mean_values[c];
			}
		}
	}

	return img;
}
template cv::Mat BlobToColorImage(const Blob<float>* blob, const int n,
		const std::vector<float> mean_values);
template cv::Mat BlobToColorImage(const Blob<double>* blob, const int n,
		const std::vector<double> mean_values);

template <typename Dtype>
void GetResizeRules(
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width,
		Blob<Dtype>* weights, 
		Blob<int>* locs,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count, 
		const int num) 
{
	CHECK_GE(coefs.size(), 2);
	CHECK(coefs.size() == (2 * num * coord_maps_count));

	// 双线性插值
	vector<int> shape(4);
	shape[0] = num;
	shape[1] = coord_maps_count;
	shape[2] = dst_height * dst_width;
	shape[3] = 4;

	weights->Reshape(shape);
	locs->Reshape(shape);

	Dtype* weights_data = weights->mutable_cpu_data();
	int* locs_data = locs->mutable_cpu_data();
	caffe::caffe_set(locs->count(), Dtype(0), weights_data);
	caffe::caffe_set(weights->count(), 0, locs_data);

	// idx： weights和locs_[i]对应的位置
	int idx = 0;
	for (int coefs_i = 0; coefs_i < coefs.size(); coefs_i += 2) {
		for (int dst_h = 0; dst_h < dst_height; ++dst_h) {
			Dtype fh = coefs[coefs_i].first * dst_h + coefs[coefs_i].second;
			const int src_h = std::floor(fh);

			fh -= src_h;
			const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
			const Dtype w_h1 = std::abs(fh);

			for (int dst_w = 0; dst_w < dst_width; ++dst_w) {
				Dtype fw = coefs[coefs_i + 1].first * dst_w + coefs[coefs_i + 1].second;
				const int src_w = std::floor(fw);

				fw -= src_w;
				const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
				const Dtype w_w1 = std::abs(fw);

				int tmp_src_h = src_h;
				int tmp_src_w = src_w;

				if (tmp_src_h >= 0 && tmp_src_h < src_height
						&& tmp_src_w >= 0 && tmp_src_w < src_width) {
					locs_data[idx] = tmp_src_h * src_width + tmp_src_w;
					weights_data[idx] = w_w0 * w_h0;
				}
				++idx;

				tmp_src_h = src_h;
				tmp_src_w = src_w + 1;
				if (tmp_src_h >= 0 && tmp_src_h < src_height
						&& tmp_src_w >= 0 && tmp_src_w < src_width) {
					locs_data[idx] = tmp_src_h * src_width + tmp_src_w;
					weights_data[idx] = w_w1 * w_h0;
				}
				++idx;

				tmp_src_h = src_h + 1;
				tmp_src_w = src_w;
				if (tmp_src_h >= 0 && tmp_src_h < src_height
						&& tmp_src_w >= 0 && tmp_src_w < src_width) {
					locs_data[idx] = tmp_src_h * src_width + tmp_src_w;
					weights_data[idx] = w_w0 * w_h1;
				}
				++idx;

				tmp_src_h = src_h + 1;
				tmp_src_w = src_w + 1;
				if (tmp_src_h >= 0 && tmp_src_h < src_height
						&& tmp_src_w >= 0 && tmp_src_w < src_width) {
					locs_data[idx] = tmp_src_h * src_width + tmp_src_w;
					weights_data[idx] = w_w1 * w_h1;
				}
				++idx;
			}
		}
	}
}
template void GetResizeRules(
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width,
		Blob<float>* weights, 
		Blob<int>* locs,
		const vector<pair<float, float> >& coefs,
		const int coord_maps_count, 
		const int num);
template void GetResizeRules(
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width,
		Blob<double>* weights, 
		Blob<int>* locs,
		const vector<pair<double, double> >& coefs,
		const int coord_maps_count, 
		const int num);

template <typename Dtype>
void AffineWarpBlob_cpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count, 
		const int num) 
{
	Blob<Dtype> weights;
	Blob<int> locs;
	GetResizeRules(src->height(), src->width(),
			dst->height(), dst->width(),
			&weights, &locs, coefs, coord_maps_count, num);

	AffineWarpBlob_cpu(src, dst, &weights, &locs);
}
template  void AffineWarpBlob_cpu(
		const Blob<float>* src,
	  Blob<float>* dst,
		const vector<pair<float, float> >& coefs,
		const int coord_maps_count, 
		const int num);
template  void AffineWarpBlob_cpu(
		const Blob<double>* src, 
		Blob<double>* dst,
		const vector<pair<double, double> >& coefs,
		const int coord_maps_count, 
		const int num);

template <typename Dtype>
void AffineWarpBlob_cpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst,
		const Blob<Dtype>* weights, 
		const Blob<int>* locs) 
{
	const Dtype* src_data = src->cpu_data();
	const int src_step = src->height() * src->width();

	Dtype* dst_data = dst->mutable_cpu_data();
	const int dst_step = dst->height() * dst->width();

	const int count = dst->count();
	const int dst_channels = dst->channels();

	const Dtype* weights_data = weights->cpu_data();
	const int* locs_data = locs->cpu_data();
	const int weight_num = weights->num();
	const int weight_channels = weights->channels();
	const int weight_height = weights->height();
	const int weight_width = weights->width();

	for (int index = 0; index < count; ++index) {
		// 以及在该height * width下的位置
		const int i = index / dst_step;
		const int j = index % dst_step;

		// 算出是第几个样本
		const int n = i / dst_channels;
		// 求出是第几个channels
		const int c = i % dst_channels;
		// 算出该channels下对应的x/y映射的offset
		const int weight_n = n % weight_num;
		const int weight_c = c / (dst_channels / weight_channels);

		const int in_offset = i * src_step;
		const int weights_offset =
				((weight_n * weight_channels + weight_c) * weight_height + j) * weight_width;

		dst_data[index] = 0;
		for (int k = 0; k < weight_width; ++k) {
			dst_data[index] = dst_data[index] +
					weights_data[weights_offset + k] *
					src_data[in_offset + locs_data[weights_offset + k]];
		}
	}
}
template void AffineWarpBlob_cpu(
		const Blob<float>* src, 
		Blob<float>* dst,
		const Blob<float>* weights, 
		const Blob<int>* locs);
template void AffineWarpBlob_cpu(
		const Blob<double>* src, 
		Blob<double>* dst,
		const Blob<double>* weights, 
		const Blob<int>* locs);

template<typename Dtype>
void CropAndResizeBlob(
		const Blob<Dtype>& src, 
		Blob<Dtype>& dst, 
		const vector<float>& coords,
		const bool is_fill, 
		const Dtype fill_value) 
{
  const int src_num = src.num();
  const int src_channels = src.channels();
  const int src_height = src.height();
  const int src_width = src.width();
  const Dtype* src_data = src.cpu_data();

	CHECK_EQ(src_num, dst.num());
	CHECK_EQ(src_channels, dst.channels());

  const int x1 = MIN(src.width() - 1, MAX(0, coords[0]));
  const int y1 = MIN(src.height() - 1, MAX(0, coords[1]));
  const int x2 = MIN(src.width() - 1, MAX(0, coords[2]));
  const int y2 = MIN(src.height() - 1, MAX(0, coords[3]));

  Blob<Dtype> tmp_dst;
	if (is_fill) {
		tmp_dst.Reshape(
				src.num(), 
				src.channels(), 
				coords[3] - coords[1] + 1, 
				coords[2] - coords[0] + 1
		);
		Dtype* tmp_dst_data = tmp_dst.mutable_cpu_data();

	  const int tmp_dst_num = tmp_dst.num();
	  const int tmp_dst_channels = tmp_dst.channels();
	  const int tmp_dst_height = tmp_dst.height();
	  const int tmp_dst_width = tmp_dst.width();

	  int tmp_dst_idx = 0;
		for (int n = 0; n < tmp_dst_num; ++n) {
			for (int c = 0; c < tmp_dst_channels; ++c) {
				for (int h = 0, src_h = coords[1]; h < tmp_dst_height; ++h, ++src_h) {
					for (int w = 0, src_w = coords[0]; w < tmp_dst_width; ++w, ++src_w, ++tmp_dst_idx) {
						if (src_h < 0  || src_h >= src_height || src_w < 0  || src_w >= src_width) {
							tmp_dst_data[tmp_dst_idx] = fill_value;
						} else {
							const int src_idx = ((n * src_channels + c) * src_height + src_h) * src_width + src_w;
							tmp_dst_data[tmp_dst_idx] = src_data[src_idx];
						}
					}
				}
			}
		}

	} else {
		tmp_dst.Reshape(src.num(), src.channels(), y2 - y2 + 1, x2 - x1 + 1);
		Dtype* tmp_dst_data = tmp_dst.mutable_cpu_data();

	  const int tmp_dst_num = tmp_dst.num();
	  const int tmp_dst_channels = tmp_dst.channels();
	  const int tmp_dst_height = tmp_dst.height();
	  const int tmp_dst_width = tmp_dst.width();

	  int tmp_dst_idx = 0;
		for (int n = 0; n < tmp_dst_num; ++n) {
			for (int c = 0; c < tmp_dst_channels; ++c) {
				for (int h = 0; h < tmp_dst_height; ++h) {
					for (int w = 0; w < tmp_dst_width; ++w, ++tmp_dst_idx) {
						const int src_idx = ((n * src_channels + c) * src_height + (h + y1)) * src_width + (w + x1);
						tmp_dst_data[tmp_dst_idx] = src_data[src_idx];
					}
				}
			}
		}
	}

	ResizeBlob_cpu(&tmp_dst, &dst);
}
template void CropAndResizeBlob(
		const Blob<double>& src, 
		Blob<double>& dst, 
		const vector<float>& coords,
		const bool is_fill, 
		const double fill_value);
template void CropAndResizeBlob(
		const Blob<float>& src, 
		Blob<float>& dst, 
		const vector<float>& coords,
		const bool is_fill, 
		const float fill_value);

cv::Mat RotateImage(cv::Mat &src, float rotation_angle)
{
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    double scale = 1;

    // Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, rotation_angle, scale);

    // Rotate the warped image
    cv::warpAffine(src, src, rot_mat, src.size());

    return rot_mat;
}
// cv::Mat RotateImage(cv::Mat &src, float rotation_angle);

} // namespace caffe