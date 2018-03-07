// Copyright 2015 Zhu.Jin Liang

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/util_pre_define.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
__global__ void kernel_BiLinearResize(
		const int nthreads, 
		const Dtype* src_data, 
		const int src_height, 
		const int src_width,
		Dtype* dst_data, 
		const int dst_height, 
		const int dst_width, 
		const Dtype scale_h, 
		const Dtype scale_w)
{
	CUDA_KERNEL_LOOP(i, nthreads) {
		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
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

template <typename Dtype>
void BiLinearResizeMat_gpu(
		const Dtype* src, 
		const int src_height, 
		const int src_width,
		Dtype* dst, 
		const int dst_height, 
		const int dst_width)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;
	kernel_BiLinearResize<Dtype> 
		<<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
			loop_n,src, src_height, src_width, dst, 
			dst_height, dst_width, scale_h, scale_w);

	//CUDA_POST_KERNEL_CHECK;
}
template void BiLinearResizeMat_gpu(
		const float* src, 
		const int src_height, 
		const int src_width,
		float* dst, 
		const int dst_height, 
		const int dst_width);

template void BiLinearResizeMat_gpu(
		const double* src, 
		const int src_height, 
		const int src_width,
		double* dst, 
		const int dst_height, 
		const int dst_width);

template <typename Dtype>
void ResizeBlob_gpu(
		const Blob<Dtype>* src, 
		const int src_n, 
		const int src_c,
		Blob<Dtype>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff /* true: data, false: diff */) 
{
	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = 
			(src_n * src_channels + src_c) * 
				src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = 
			(dst_n * dst_channels + dst_c) * 
				dst_height * dst_width;

	// const Dtype scale_w = src_width / (Dtype)dst_width;
	// const Dtype scale_h = src_height / (Dtype)dst_height;
	
	const Dtype* src_data = 
			data_or_diff ?
			&(src->gpu_data()[src_offset]) :
			&(src->gpu_diff()[src_offset]);

	Dtype* dst_data = 
			data_or_diff ? 
			&(dst->mutable_gpu_data()[dst_offset]) :
			&(dst->mutable_gpu_diff()[dst_offset]);

	BiLinearResizeMat_gpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
	CUDA_POST_KERNEL_CHECK;
}
template void ResizeBlob_gpu(
		const Blob<float>* src, 
		const int src_n, 
		const int src_c,
		Blob<float>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff);
template void ResizeBlob_gpu(
		const Blob<double>* src, 
		const int src_n, 
		const int src_c,
		Blob<double>* dst, 
		const int dst_n, 
		const int dst_c,
		const bool data_or_diff);

template <typename Dtype>
__global__ void kernel_GetBiLinearResizeMatRules(
		const int nthreads,  
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width, 
		const Dtype scale_h, 
		const Dtype scale_w,
		Dtype* loc1, 
		Dtype* weight1, 
		Dtype* loc2, 
		Dtype* weight2,
		Dtype* loc3, 
		Dtype* weight3, 
		Dtype* loc4, 
		Dtype* weight4)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		int dst_h = index /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = index %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
		// dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		loc1[dst_idx] = src_idx;
		weight1[dst_idx] = w_h0 * w_w0;

		if (src_w + 1 < src_width)
		{
			loc2[dst_idx] = src_idx + 1;
			weight2[dst_idx] = w_h0 * w_w1;
			// dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
		}

		if (src_h + 1 < src_height)
		{
			// dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
			weight3[dst_idx] = w_h1 * w_w0;
			loc3[dst_idx] = src_idx + src_width;
		}

		if (src_w + 1 < src_width && src_h + 1 < src_height)
		{
			loc4[dst_idx] = src_idx + src_width + 1;
			weight4[dst_idx] = w_h1 * w_w1;
			// dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
		}
	}
}


template <typename Dtype>
__global__ void kernel_ResizeBlob(
		const int nthreads,
		const int num,
		const int channels, 
		const Dtype* src, 
		const int src_height, 
		const int src_width,
		Dtype* dst, 
		const int dst_height, 
		const int dst_width, 
		const Dtype scale_h, 
		const Dtype scale_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index %( dst_height * dst_width);
		int c = (index/(dst_height * dst_width))%channels;
		int n = (index/(dst_height * dst_width))/channels;
		int src_offset = (n * channels + c) * src_height * src_width;
		int dst_offset = (n * channels + c) * dst_height * dst_width;

		const Dtype* src_data = src+src_offset;
		Dtype* dst_data = dst+dst_offset;

		int dst_h = i /dst_width;
		Dtype fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const Dtype w_h0 = std::abs(1.0f - fh);
		const Dtype w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		Dtype fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const Dtype w_w0 = std::abs(1.0f - fw);
		const Dtype w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;
		dst_data[dst_idx] = 0;

		const int src_idx = src_offset_1 + src_w;

		dst_data[dst_idx] += 
				(w_h0 * w_w0 * src_data[src_idx]);

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

template <typename Dtype>
void ResizeBlob_gpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst, 
		/* true: data, false: diff */
		const bool data_or_diff) 
{
	CHECK(src->num() == dst->num())
			<<"src->num() == dst->num()";
	CHECK(src->channels() == dst->channels())
			<< "src->channels() == dst->channels()";

	const int src_num = src->num();
	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();


	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();


	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;
	int loop_n = dst_height * dst_width*dst_channels*src_num;

	const Dtype* src_data = 
			data_or_diff ? 
			src->gpu_data() :
			src->gpu_diff();
	Dtype* dst_data = 
			data_or_diff ? 
			dst->mutable_gpu_data() :
			dst->mutable_gpu_diff();

	kernel_ResizeBlob<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), 
		CAFFE_CUDA_NUM_THREADS >>>
			(loop_n,src_num,src_channels,
				src_data, src_height,src_width,
				dst_data, dst_height, dst_width,
				scale_h,scale_w);
	CUDA_POST_KERNEL_CHECK;
}
template void ResizeBlob_gpu(
		const Blob<float>* src,
		Blob<float>* dst,
		const bool data_or_diff);
template void ResizeBlob_gpu(
		const Blob<double>* src,
		Blob<double>* dst,
		const bool data_or_diff);

template <typename Dtype>
void ResizeBlob_Data_gpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst)
{
	const bool data_or_diff = true;
	ResizeBlob_gpu(src, dst, data_or_diff);
}
template void ResizeBlob_Data_gpu(
		const Blob<float>* src,
		Blob<float>* dst);
template void ResizeBlob_Data_gpu(
		const Blob<double>* src,
		Blob<double>* dst);

template <typename Dtype>
void ResizeBlob_Diff_gpu(
		const Blob<Dtype>* src, 
		Blob<Dtype>* dst)
{
	const bool data_or_diff = false;
	ResizeBlob_gpu(src, dst, data_or_diff);
}
template void ResizeBlob_Diff_gpu(
		const Blob<float>* src,
		Blob<float>* dst);
template void ResizeBlob_Diff_gpu(
		const Blob<double>* src,
		Blob<double>* dst);

template <typename Dtype>
void GetBiLinearResizeMatRules_gpu( 
		const int src_height, 
		const int src_width,
		const int dst_height, 
		const int dst_width,
		Dtype* loc1, 
		Dtype* weight1, 
		Dtype* loc2, 
		Dtype* weight2,
		Dtype* loc3, 
		Dtype* weight3, 
		Dtype* loc4, 
		Dtype* weight4)
{
	const Dtype scale_w = src_width / (Dtype)dst_width;
	const Dtype scale_h = src_height / (Dtype)dst_height;


	int loop_n = dst_height * dst_width;
	caffe::caffe_gpu_set(loop_n,(Dtype)0,loc1);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,loc2);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,loc4);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,loc3);

	caffe::caffe_gpu_set(loop_n,(Dtype)0,weight1);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,weight2);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,weight3);
	caffe::caffe_gpu_set(loop_n,(Dtype)0,weight4);
	kernel_GetBiLinearResizeMatRules<Dtype> 
		<<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
			loop_n,  src_height,  src_width,
			dst_height, dst_width, scale_h, scale_w,
			loc1,  weight1,  loc2,  weight2,
			loc3,  weight3,   loc4,   weight4);
	CUDA_POST_KERNEL_CHECK;
}
template void GetBiLinearResizeMatRules_gpu(  
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
template void GetBiLinearResizeMatRules_gpu(  
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
void ResizeBlob_gpu(
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

	GetBiLinearResizeMatRules_gpu(
			src->height(),
			src->width(),
			dst->height(), 
			dst->width(),
			loc1->mutable_gpu_data(), 
			loc1->mutable_gpu_diff(), 
			loc2->mutable_gpu_data(), 
			loc2->mutable_gpu_diff(),
			loc3->mutable_gpu_data(), 
			loc3->mutable_gpu_diff(), 
			loc4->mutable_gpu_data(), 
			loc4->mutable_gpu_diff());

	for(int n=0;n< src->num();++n)
	{
		for(int c=0; c < src->channels() ; ++c)
		{
			ResizeBlob_gpu(src,n,c,dst,n,c);
		}
	}
}
template void ResizeBlob_gpu(
		const Blob<float>* 
		src,Blob<float>* dst,
		Blob<float>* loc1, 
		Blob<float>* loc2, 
		Blob<float>* loc3, 
		Blob<float>* loc4);
template void ResizeBlob_gpu(
		const Blob<double>* src,
		Blob<double>* dst,
		Blob<double>* loc1, 
		Blob<double>* loc2, 
		Blob<double>* loc3, 
		Blob<double>* loc4);

template <typename Dtype>
__global__ void affine_warp_blob_kernel(const int nthreads,
		const Dtype* src_data, const int src_step,
		Dtype* dst_data, const int dst_step, const int dst_channels,
		const Dtype* weights, const int* locs,
		const int weight_num, const int weight_channels,
		const int weight_height, const int weight_width) {

  CUDA_KERNEL_LOOP(index, nthreads) {

  	// 把num跟channels合成一个维度
  	// 算出当前是第几个height * width
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
  	const int weights_offset = ((weight_n * weight_channels + weight_c) * weight_height + j) * weight_width;

  	dst_data[index] = 0;
  	for (int k = 0; k < weight_width; ++k) {
  		dst_data[index] += (weights[weights_offset + k] * src_data[in_offset + locs[weights_offset + k]]);
  	}
  }
}

template <typename Dtype>
void AffineWarpBlob_gpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const vector<pair<Dtype, Dtype> >& coefs,
		const int coord_maps_count, const int num) {

	Blob<Dtype> weights;
	Blob<int> locs;
	GetResizeRules(src->height(), src->width(),
			dst->height(), dst->width(),
			&weights, &locs, coefs, coord_maps_count, num);

	AffineWarpBlob_gpu(src, dst, &weights, &locs);
}

template void AffineWarpBlob_gpu(const Blob<float>* src, Blob<float>* dst,
		const vector<pair<float, float> >& coefs,
		const int coord_maps_count, const int num);
template void AffineWarpBlob_gpu(const Blob<double>* src, Blob<double>* dst,
		const vector<pair<double, double> >& coefs,
		const int coord_maps_count, const int num);

template <typename Dtype>
void AffineWarpBlob_gpu(const Blob<Dtype>* src, Blob<Dtype>* dst,
		const Blob<Dtype>* weights, const Blob<int>* locs) {

	const Dtype* src_data = src->gpu_data();
	const int src_step = src->height() * src->width();

	Dtype* dst_data = dst->mutable_gpu_data();
	const int dst_step = dst->height() * dst->width();

	const int count = dst->count();

	affine_warp_blob_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count,
			src_data, src_step,
			dst_data, dst_step, dst->channels(),
			weights->gpu_data(), locs->gpu_data(),
			weights->num(), weights->channels(),
			weights->height(), weights->width());
	CUDA_POST_KERNEL_CHECK;

}

template void AffineWarpBlob_gpu(const Blob<float>* src, Blob<float>* dst,
		const Blob<float>* weights, const Blob<int>* locs);
template void AffineWarpBlob_gpu(const Blob<double>* src, Blob<double>* dst,
		const Blob<double>* weights, const Blob<int>* locs);

} // namespace caffe