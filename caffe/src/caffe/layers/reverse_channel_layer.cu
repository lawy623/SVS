
#include <vector>

#include "caffe/layers/reverse_channel_layer.hpp"
#include "caffe/util/util_img.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/util/StereoEngine.hpp"
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iomanip>

namespace caffe {


template <typename Dtype>
void ReverseChannelLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int num = bottom[0] -> num();
    const int channels = bottom[0] -> channels();
    const int height = bottom[0] -> height();
    const int width = bottom[0] -> width();
    const int size = height * width;
    const Dtype *bottom_data = bottom[0] -> gpu_data();
    Dtype *top_data = top[0] -> mutable_gpu_data();

    for(int n = 0; n < num; ++n){
        for(int c = 0; c < channels; ++c){
            int bottom_offset = bottom[0] -> offset(n,c);
            int top_offset = top[0] -> offset(n, (channels - c - 1));
            caffe_gpu_memcpy(size*sizeof(Dtype), bottom_data + bottom_offset, top_data + top_offset);
        }
    }
}
template <typename Dtype>
void ReverseChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    const int num = bottom[0] -> num();
    const int channels = bottom[0] -> channels();
    const int height = bottom[0] -> height();
    const int width = bottom[0] -> width();
    const int size = height * width;
    Dtype *bottom_diff = bottom[0] -> mutable_gpu_diff();
    const Dtype *top_diff = top[0] -> gpu_diff();

    if(propagate_down[0]){
        for(int n = 0; n < num; ++n){
            for(int c = 0; c < channels; ++c){
                int bottom_offset = bottom[0] -> offset(n,c);
                int top_offset = top[0] -> offset(n, (channels - c - 1));
                caffe_gpu_memcpy(size*sizeof(Dtype), top_diff + top_offset, bottom_diff + bottom_offset);
            }
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseChannelLayer);

}  // namespace caffe