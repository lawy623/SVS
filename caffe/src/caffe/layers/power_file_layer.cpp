#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/power_file_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerFileLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tPower File Layer:: LayerSetUp: \t";

	vector<int> shift_shape(1);
	shift_shape[0] = bottom[0]->count(1);
	shift_.Reshape(shift_shape);

	Dtype* shift = shift_.mutable_cpu_data();
	caffe_set(shift_.count(), (Dtype)0, shift);

	if(this->layer_param_.power_file_param().has_shift_file()) {
		string filename = this->layer_param_.power_file_param().shift_file();
		std::cout << prefix << "Reading shift file " + filename << std::endl;
		std::ifstream fin(filename.c_str());
		Dtype tmp; int k = 0;
		while(fin >> tmp && k < bottom[0]->count(1)) {
			shift[k] = tmp;
			++k;
		}
		fin.close();
	} else {
		std::cout << prefix << "No shift file detected. Intialize all to zero." << std::endl;
	}

	std::cout << prefix << "Setting shift vector to: ";
	for(int i=0; i<shift_.count(); ++i) {
		std::cout << shift[i] << "\t";
	}
	std::cout << std::endl;

	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void PowerFileLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	top[0]->ReshapeLike(*bottom[0]);

	vector<int> shift_shape(1);
	shift_shape[0] = bottom[0]->count(1);
	shift_.Reshape(shift_shape);
}

template <typename Dtype>
void PowerFileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

template <typename Dtype>
void PowerFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	/* not implemented */
	CHECK(false) << "Error: not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(PowerFileLayer);
#endif

INSTANTIATE_CLASS(PowerFileLayer);
REGISTER_LAYER_CLASS(PowerFile);

}  // namespace caffe
