#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/power_file_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PowerFileForwardGPU(const int nthreads, const int size, 
	const Dtype* input, const Dtype* shift, Dtype* output) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {
		
		const int shift_idx = index % size;
		output[index] = input[index] + shift[shift_idx];
	}
}

template <typename Dtype>
void PowerFileLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const Dtype* shift = shift_.gpu_data();
	const Dtype* input = bottom[0]->gpu_data();
	Dtype* output = top[0]->mutable_gpu_data();
	
	CHECK(top[0]->count() == bottom[0]->count()) << "Error: in Forward_gpu of PowerFileLayer.";
	CHECK(bottom[0]->count() % shift_.count() == 0) << "Error: in Forward_gpu of PowerFileLayer.";
	
	const int nthreads = bottom[0]->count();
	PowerFileForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, shift_.count(), input, shift, output);
}

template <typename Dtype>
__global__ void PowerFileBackwardGPU(const int nthreads, 
	const Dtype* output, Dtype* input) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {
		
		input[index] = output[index];
	}
}

template <typename Dtype>
void PowerFileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* output = top[0]->gpu_diff();
	Dtype* input = bottom[0]->mutable_gpu_diff();
	
	CHECK(top[0]->count() == bottom[0]->count()) << "Error: in Backward_gpu of PowerFileLayer.";
	
	const int nthreads = bottom[0]->count();
	PowerFileBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	     CAFFE_CUDA_NUM_THREADS>>>(nthreads, output, input);
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerFileLayer);

}	// namespace caffe
