#include <vector>
#include <iostream>
#include "caffe/layers/remap_layer.hpp"

namespace caffe {

template<typename Dtype>
void RemapLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                      const vector< Blob< Dtype >* >& top) {
	// Nothing to setup for
}

template<typename Dtype>
void RemapLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                   const vector< Blob< Dtype >* >& top) {
	CHECK_EQ(bottom.size(), 2) << "Remap only supports two bottoms";
	CHECK_EQ(top.size(), 1) << "Remap only supports one top";
	CHECK_EQ(bottom[0]->num(), 
		bottom[1]->num()) << "Remap requires num to be the same for both bottom blobs";
	CHECK_EQ(bottom[1]->channels(), 2) << "Remap requires coords blob (bottom[1]) to have only 2 channels";
	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 
		bottom[1]->height(), bottom[1]->width());
}

template<typename Dtype>
inline Dtype getvalue(Dtype* V, int x, int y, int c, int n, int C, int W, int H) {
	if ( x < 0 || x > W - 1 || y < 0 || y > H - 1 ) 
		return 0;	
	return V[((n*C+c)*H+y)*W+x];
}

template<typename Dtype>
inline void addvalue(Dtype* V, int x, int y, int c, int n, int C, int W, int H, Dtype v) {
	if ( x < 0 || x > W - 1 || y < 0 || y > H - 1 ) 
		return;
	V[((n*C+c)*H+y)*W+x] += v;
}

template <typename Dtype>
void RemapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    								const vector<Blob<Dtype>*>& top) {
	const Dtype* Vb = bottom[0]->cpu_data();
	const Dtype* coords = bottom[1]->cpu_data();
	Dtype* Vt = top[0]->mutable_cpu_data();
	int N  = bottom[0]->num();
	int C  = bottom[0]->channels();
	int W0 = bottom[0]->width();
	int H0 = bottom[0]->height();
	int W1 = bottom[1]->width();
	int H1 = bottom[1]->height();
	float x, y, wx, wy, w00, w01, w10, w11, v00, v01, v10, v11;
	int x0, y0, x1, y1;
	for ( int n = 0; n < N; n++ ) {
		for ( int c = 0; c < C; c++ ) {
			for ( int h = 0; h < H1; h++ ) {
				for ( int w = 0; w < W1; w++ ) {
					x = coords[((n*2+0)*H1+h)*W1+w];
					y = coords[((n*2+1)*H1+h)*W1+w];
					x0 = floor(x);
					y0 = floor(y);
					x1 = x0 + 1;
					y1 = y0 + 1;
					wx = x - x0;
					wy = y - y0;
					w00 = (1 - wx) * (1 - wy);
					w01 = (1 - wx) * wy;
					w10 = wx * (1 - wy);
					w11 = wx * wy;
					v00 = getvalue(Vb, x0, y0, c, n, C, W0, H0);
					v01 = getvalue(Vb, x0, y1, c, n, C, W0, H0);
					v10 = getvalue(Vb, x1, y0, c, n, C, W0, H0);
					v11 = getvalue(Vb, x1, y1, c, n, C, W0, H0);
					Vt[((n*C+c)*H1+h)*W1+w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
				}
			}
		}
	} 
}

template <typename Dtype>
void RemapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    								const vector<bool>& propagate_down,
    								const vector<Blob<Dtype>*>& bottom) {

	const Dtype* Vb = bottom[0]->cpu_data();
	const Dtype* coords = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();

	// FIXME: necessary?
	if (propagate_down[0]) {
		caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
	}

	if (propagate_down[1]) {
		caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
	}
	
	int N = bottom[0]->num();
	int C = bottom[0]->channels();
	int W0 = bottom[0]->width();
	int H0 = bottom[0]->height();
	int W1 = bottom[1]->width();
	int H1 = bottom[1]->height();

	float x, y, wx, wy, v00, v01, v10, v11;
	float dx, dy, dv00, dv01, dv10, dv11;
	int x0, y0, x1, y1;
	for ( int n = 0; n < N; n++ ) {
		for ( int c = 0; c < C; c++ ) {
			for ( int h = 0; h < H1; h++ ) {
				for ( int w = 0; w < W1; w++ ) {
					x = coords[((n*2+0)*H1+h)*W1+w];
					y = coords[((n*2+1)*H1+h)*W1+w];
					x0 = floor(x);
					y0 = floor(y);
					x1 = x0 + 1;
					y1 = y0 + 1;
					wx = x - x0;
					wy = y - y0;
					v00 = getvalue(Vb, x0, y0, c, n, C, W0, H0);
					v01 = getvalue(Vb, x0, y1, c, n, C, W0, H0);
					v10 = getvalue(Vb, x1, y0, c, n, C, W0, H0);
					v11 = getvalue(Vb, x1, y1, c, n, C, W0, H0);
					// Gradients
					dx = (wy-1)*v00 - wy*v01 + (1-wy)*v10 + wy*v11;
					dy = (wx-1)*v00 - wx*v10 + (1-wx)*v01 + wx*v11;
					dv00 = (1-wx)*(1-wy);
					dv01 = (1-wx)*wy;
					dv10 = wx*(1-wy);
					dv11 = wx*wy;
					if (propagate_down[0]) {
						Dtype* b0_diff = bottom[0]->mutable_cpu_diff();
						// Backprop for bottom[0] (values)
						addvalue(b0_diff, x0, y0, c, n, C, W0, H0, dv00 * top_diff[((n*C+c)*H1+h)*W1+w]);
						addvalue(b0_diff, x0, y1, c, n, C, W0, H0, dv01 * top_diff[((n*C+c)*H1+h)*W1+w]);
						addvalue(b0_diff, x1, y0, c, n, C, W0, H0, dv10 * top_diff[((n*C+c)*H1+h)*W1+w]);
						addvalue(b0_diff, x1, y1, c, n, C, W0, H0, dv11 * top_diff[((n*C+c)*H1+h)*W1+w]);
					}
					if (propagate_down[1]) {
						// Backprop for bottom[1] (coords)
						Dtype* b1_diff = bottom[1]->mutable_cpu_diff();
						addvalue(b1_diff, w,  h,  0, n, 2, W1, H1, dx * top_diff[((n*C+c)*H1+h)*W1+w]);
						addvalue(b1_diff, w,  h,  1, n, 2, W1, H1, dy * top_diff[((n*C+c)*H1+h)*W1+w]);
					}
				}
			}
		} 
	}
}

#ifdef CPU_ONLY
STUB_GPU(RemapLayer);
#endif

INSTANTIATE_CLASS(RemapLayer);
REGISTER_LAYER_CLASS(Remap);

}  // namespace caffe