#ifndef STEREOENGINE_HPP_
#define STEREOENGINE_HPP_


namespace caffe {

#ifndef TEXTURE_CONSTANT
#define TEXTURE_CONSTANT
 texture <uchar, 2, cudaReadModeElementType> ltex;
 texture <uchar, 2, cudaReadModeElementType> rtex;
 texture <uchar, 2, cudaReadModeElementType> ltex1;
 texture <uchar, 2, cudaReadModeElementType> rtex1;
 texture <short, 2, cudaReadModeElementType> short_tex0;
 texture <short, 2, cudaReadModeElementType> short_tex1;
 texture <short, 2, cudaReadModeElementType> short_tex2;
 texture <short, 2, cudaReadModeElementType> short_tex3;
 __constant__ uchar table[2304];

#endif



#define gpuErrchk(ans) { gpuAssert(ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename DtypeA, typename DtypeB>
__global__ void CopyDtypeAToDtypeB(const int n_elements, const DtypeA *a_data, DtypeB *b_data);


__global__ void init_dTable(uchar* d_table, const int TAB_OFS, const int tableSize, int ftzero);

__global__ void truncated_sobel_kernel(uchar *d_lSobel, uchar *d_rSobel, int width, int height, int TAB_OFS);

__device__ void costAtDispImg_v3(int* sum0, int* sum1, int x, int y, int d);

__global__ void cost_Pix_BT_img_v3(short *d_CPixbuf, int width, int height, int img_width, int minD,  int maxD,
                                    float sobel_multiply_scale, float img_multiply_scale,
                                    int scale_h, int scale_w, int step_mode);

//box filter -- sum
__global__ void cost_Block_BT(short* d_out, const int width, const int height, const int SW, const int SH, const int P2);


__host__ void computeBTCostVolume(const unsigned char*grayleft, const unsigned char *grayright,
        int img_width, int img_height, short *cost_volume,
        int disp_width, int disp_height, int min_disp, int disp_num, int step_h, int step_w,
        float sobel_multiply_scale, float img_multiply_scale, int step_mode);

__host__ void computeSGBM(const short*cost_volume,
        int img_width, int img_height, short *disp1, unsigned char *udispconf,
        int disp_width, int disp_height, int min_disp, int disp_num);

void disp2color(const float *fdisp, unsigned char *color, int width, int height, int minDisparity, int numDisparities) ;

} // namespace caffe
#endif //STEREOENGINE_HPP_
