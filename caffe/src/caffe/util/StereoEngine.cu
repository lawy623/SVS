#include <iostream>
#include <fstream>
#ifndef uchar
#define uchar unsigned char
#endif

#include "caffe/util/StereoEngine.hpp"


// #include "cudaUtils.h"
#include "caffe/util/sgbm/calc_S4_3_v6.h"
#include "caffe/util/sgbm/calc_S4_2_v1.h"
#include "caffe/util/sgbm/calc_S4_1_v3.h"
#include "caffe/util/sgbm/calc_D_v0.h"
#include "caffe/util/sgbm/transpose.cuh"

#include "caffe/common.hpp"
#include <math.h>
#include <cuda.h>

namespace caffe {


typedef uchar PixType;
typedef short CostType;
typedef short DispType;


template <typename DtypeA, typename DtypeB>
__global__ void CopyDtypeAToDtypeB(const int n_elements, const DtypeA *a_data, DtypeB *b_data)
{
    CUDA_KERNEL_LOOP(index, n_elements){
        b_data[index] = (DtypeB)a_data[index];
    }
}
// explicit instanitiate
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const short *a_data, float *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const short *a_data, double *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const uchar *a_data, float *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const uchar *a_data, double *b_data);

template __global__ void CopyDtypeAToDtypeB(const int n_elements, const float *a_data, short *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const double *a_data, short *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const float *a_data, uchar *b_data);
template __global__ void CopyDtypeAToDtypeB(const int n_elements, const double *a_data, uchar *b_data);



__host__ void computeBTCostVolume(const unsigned char*grayleft, const unsigned char *grayright,
        int img_width, int img_height, short *cost_volume,
        int disp_width, int disp_height, int min_disp, int disp_num, int step_h, int step_w,
        float sobel_multiply_scale, float img_multiply_scale, int step_mode){



    int minD = min_disp;
    int maxD = minD + disp_num; //64; //atio(argv[2]);
    int SADWindowSize_width = 3;
    int SADWindowSize_height = 3;
    int ftzero = 63;
    int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;

    int P1 = 8*SADWindowSize_height*SADWindowSize_width, P2 = 4*P1;

    int width_ori = img_width, height_ori = img_height;
    int width = disp_width, height = disp_height;
    int minX1 = 0, maxX1 = width;
    int D = maxD - minD, width1 = maxX1 - minX1;

    int SW2 = SADWindowSize_width/2, SH2 = SADWindowSize_height/2;

    size_t costBufSize = width1 * D;
    size_t CSBufSize = costBufSize*height;

    // cudaEvent_t timeStartEvent, timeEndEvent1;
    // gpuErrchk(cudaEventCreate(&timeStartEvent, 0));
    // gpuErrchk(cudaEventCreate(&timeEndEvent1, 0));

    CostType *d_CCSbuf, *d_Cbuf, *d_Sbuf, *d_Lrbuf;
    gpuErrchk(cudaMalloc((void**) &d_CCSbuf, CSBufSize*sizeof(CostType)));
    gpuErrchk(cudaMalloc((void**) &d_Cbuf, CSBufSize*sizeof(CostType)));
    gpuErrchk(cudaMalloc((void**) &d_Sbuf, CSBufSize*sizeof(CostType)));
    gpuErrchk(cudaMalloc((void**) &d_Lrbuf, ((width1+2)*2+D*(width1+2)*2)*3*sizeof(CostType)));
    gpuErrchk(cudaMemset(d_Lrbuf, 0, ((width1+2)*2+D*(width1+2)*2)*3*sizeof(CostType)));

    gpuErrchk(cudaMemset(d_CCSbuf, 0, CSBufSize*sizeof(CostType)));
    gpuErrchk(cudaMemset(d_Cbuf, 0, CSBufSize*sizeof(CostType)));
    gpuErrchk(cudaMemset(d_Sbuf, 0, CSBufSize*sizeof(CostType)));

    // gpuErrchk(cudaEventRecord(timeStartEvent, 0));

    PixType *d_lSobel, *d_rSobel, *d_tableBuf;
    gpuErrchk(cudaMalloc((void**) &d_lSobel, width_ori*height_ori*sizeof(PixType)));
    gpuErrchk(cudaMalloc((void**) &d_rSobel, width_ori*height_ori*sizeof(PixType)));
    gpuErrchk(cudaMalloc((void**) &d_tableBuf, TAB_SIZE*sizeof(PixType)));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<PixType>();
    cudaArray *d_left, *d_right;
    gpuErrchk(cudaMallocArray(&d_left, &channelDesc, width_ori, height_ori));
    gpuErrchk(cudaMallocArray(&d_right, &channelDesc, width_ori, height_ori));
    gpuErrchk(cudaMemcpyToArray(d_left, 0, 0, grayleft, width_ori*height_ori*sizeof(PixType),   cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpyToArray(d_right, 0, 0, grayright, width_ori*height_ori*sizeof(PixType), cudaMemcpyDeviceToDevice));



    ltex.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    ltex.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    rtex.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    rtex.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    ltex1.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    ltex1.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    rtex1.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    rtex1.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;

    short_tex0.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex0.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex1.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex1.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex2.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex2.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex3.addressMode[0] = cudaAddressModeClamp; //cudaAddressModeWrap;
    short_tex3.addressMode[1] = cudaAddressModeClamp; //cudaAddressModeWrap;

    // Bind gray image to texture
    gpuErrchk(cudaBindTextureToArray(ltex, d_left));
    gpuErrchk(cudaBindTextureToArray(rtex, d_right));


    // init table for sobel filter
    init_dTable<<<divUp(TAB_SIZE, 256), 256>>>(d_tableBuf, TAB_OFS, TAB_SIZE, ftzero);
    gpuErrchk(cudaMemcpyToSymbol(table, d_tableBuf, TAB_SIZE*sizeof(PixType)));

    // Sobel filtering
    dim3 Sobelblock(32, 32);
    dim3 Sobelgrid(divUp(width_ori, 32), divUp(height_ori, 32)); //int numthreads1 = D;
    truncated_sobel_kernel<<<Sobelgrid, Sobelblock>>>(d_lSobel, d_rSobel, width_ori, height_ori, TAB_OFS);



    dim3 PixBTblock(32, 32);
    dim3 PixBTgrid(divUp(width, 32), divUp(height, 32)); //int numthreads1 = D;

    cudaArray *d_left_sobel, *d_right_sobel;
    gpuErrchk(cudaMallocArray(&d_left_sobel, &channelDesc, width_ori, height_ori));
    gpuErrchk(cudaMallocArray(&d_right_sobel, &channelDesc, width_ori, height_ori));
    gpuErrchk(cudaMemcpyToArray(d_left_sobel, 0, 0, d_lSobel, width_ori*height_ori*sizeof(PixType), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpyToArray(d_right_sobel, 0, 0, d_rSobel, width_ori*height_ori*sizeof(PixType), cudaMemcpyDeviceToDevice));

    // Bind sobel
    gpuErrchk(cudaBindTextureToArray(ltex1, d_left_sobel));
    gpuErrchk(cudaBindTextureToArray(rtex1, d_right_sobel));


    // compute pixel BT cost
    cost_Pix_BT_img_v3<<<PixBTgrid, PixBTblock>>>(d_CCSbuf, width, height, width_ori,
            minD, maxD, sobel_multiply_scale, img_multiply_scale , step_h,
            step_w, step_mode);

    cudaUnbindTexture(ltex); cudaUnbindTexture(rtex);
    cudaUnbindTexture(ltex1); cudaUnbindTexture(rtex1);



    cudaDeviceSynchronize();
    /***********************calculate block cost matrix C *********************************/
    // NOTE:  D % 4 == 0 !!!!!!!!!!!!!!!!
    cudaChannelFormatDesc channelDesc_short = cudaCreateChannelDesc<CostType>();
    dim3 BlockBTblock(16,16);
    dim3 BlockBTgrid(divUp(width, 16), divUp(height, 16)); //int numthreads1 = D;


    cudaArray *d_short_tex0_array, *d_short_tex1_array,*d_short_tex2_array,*d_short_tex3_array;
    gpuErrchk(cudaMallocArray(&d_short_tex0_array, &channelDesc_short, width, height));
    gpuErrchk(cudaMallocArray(&d_short_tex1_array, &channelDesc_short, width, height));
    gpuErrchk(cudaMallocArray(&d_short_tex2_array, &channelDesc_short, width, height));
    gpuErrchk(cudaMallocArray(&d_short_tex3_array, &channelDesc_short, width, height));



    // CostType
    for(int d_idx = minD; d_idx < maxD; d_idx += 4){
        gpuErrchk(cudaMemcpyToArray(d_short_tex0_array, 0, 0, d_CCSbuf + (d_idx - minD + 0)*width*height, width*height*sizeof(CostType), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpyToArray(d_short_tex1_array, 0, 0, d_CCSbuf + (d_idx - minD + 1)*width*height, width*height*sizeof(CostType), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpyToArray(d_short_tex2_array, 0, 0, d_CCSbuf + (d_idx - minD + 2)*width*height, width*height*sizeof(CostType), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpyToArray(d_short_tex3_array, 0, 0, d_CCSbuf + (d_idx - minD + 3)*width*height, width*height*sizeof(CostType), cudaMemcpyDeviceToDevice));

        gpuErrchk(cudaBindTextureToArray(short_tex0, d_short_tex0_array));
        gpuErrchk(cudaBindTextureToArray(short_tex1, d_short_tex1_array));
        gpuErrchk(cudaBindTextureToArray(short_tex2, d_short_tex2_array));
        gpuErrchk(cudaBindTextureToArray(short_tex3, d_short_tex3_array));

        cost_Block_BT<<<BlockBTgrid, BlockBTblock>>>(d_Cbuf + (d_idx - minD + 0)*width*height, width, height, SW2, SH2, P2);
    }

    gpuErrchk(cudaUnbindTexture(short_tex0)); gpuErrchk(cudaUnbindTexture(short_tex1));
    gpuErrchk(cudaUnbindTexture(short_tex2)); gpuErrchk(cudaUnbindTexture(short_tex3));


    /*********************END*****************************/
    gpuErrchk(cudaMemcpy(cost_volume, d_Cbuf, CSBufSize*sizeof(CostType), cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaFreeArray(d_left));
    gpuErrchk(cudaFreeArray(d_right));
    gpuErrchk(cudaFreeArray(d_left_sobel));
    gpuErrchk(cudaFreeArray(d_right_sobel));
    gpuErrchk(cudaFreeArray(d_short_tex0_array));
    gpuErrchk(cudaFreeArray(d_short_tex1_array));
    gpuErrchk(cudaFreeArray(d_short_tex2_array));
    gpuErrchk(cudaFreeArray(d_short_tex3_array));


    cudaFree(d_lSobel); cudaFree(d_rSobel); cudaFree(d_tableBuf);
    cudaFree(d_Cbuf); cudaFree(d_Sbuf); cudaFree(d_Lrbuf);
    cudaFree(d_CCSbuf);


    // cudaEventRecord(timeEndEvent1, 0);
    // cudaEventSynchronize(timeEndEvent1);
    // float gpu_t1;
    // cudaEventElapsedTime(&gpu_t1, timeStartEvent, timeEndEvent1);

    return;
}




__global__ void init_dTable(uchar* d_table, const int TAB_OFS, const int tableSize, int ftzero){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if( k<tableSize ){
        d_table[k] = (uchar) (min(max(k-TAB_OFS, -ftzero), ftzero) + ftzero);

    }
}

__global__ void truncated_sobel_kernel(uchar *d_lSobel, uchar *d_rSobel, int width, int height, int TAB_OFS){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height){

        if( x == 0 || x == width-1){
            d_lSobel[y*width + x] = table[TAB_OFS];
            d_rSobel[y*width + x] = table[TAB_OFS];
        }else{

            uchar lx0, lx1, lx2, lx3, lx4, lx5, rx0, rx1, rx2, rx3, rx4, rx5;
            int s1 = y > 0 ? 1 : 0, s2 = y < height -1 ? 1 : 0;

            lx0 = tex2D(ltex, x-1, y-s1); //y-1);
            lx1 = tex2D(ltex, x-1, y);
            lx2 = tex2D(ltex, x-1, y+s2); //y+1);
            lx3 = tex2D(ltex, x+1, y-s1); //y-1);
            lx4 = tex2D(ltex, x+1, y);
            lx5 = tex2D(ltex, x+1, y+s2); //y+1);
            int lDevx = (lx5 + 2*lx4 + lx3) - (lx2 + 2*lx1 + lx0); //sobel operator at the x direction
            d_lSobel[y*width + x] = table[lDevx + TAB_OFS];

            rx0 = tex2D(rtex, x-1, y-s1);
            rx1 = tex2D(rtex, x-1, y);
            rx2 = tex2D(rtex, x-1, y+s2);
            rx3 = tex2D(rtex, x+1, y-s1);
            rx4 = tex2D(rtex, x+1, y);
            rx5 = tex2D(rtex, x+1, y+s2);
            int rDevx = (rx5 + 2*rx4 + rx3) - (rx2 + 2*rx1 + rx0); //sobel operator at the x direction
            d_rSobel[y*width + x] = table[rDevx + TAB_OFS];

        }
    }
}

// NOTE: could not merge v3_0 with v3_1
__device__ void costAtDispImg_v3_0(int* sum0, int x, int y, int d)
{
    //computation on the Image domain
    const int lI = tex2D(ltex, x, y);
    const int rI = tex2D(rtex, x - d, y);
    //NOTE: d >= 0 !!!
    const int laI = (lI + tex2D(ltex, x-1, y))/2;  //x > 0 ? (lI + tex2D(ltex, x - 1, y))/2 : lI;
    const int lbI = (lI + tex2D(ltex, x+1, y))/2; //x < width-1 ? (lI + tex2D(ltex, x + 1, y))/2 : lI;
    const int raI = (rI + tex2D(rtex, x-d-1, y))/2;  //x > d ? 0.5f*(rI + tex2D(rtex, x - d - 1, y)) : rI;
    const int rbI = (rI + tex2D(rtex, x-d+1, y))/2; //x < width+d-1 ?  (rI + tex2D(rtex, x - d + 1, y))/2 : rI;
    const int lImin = min(laI, min(lbI, lI));
    const int lImax = max(laI, max(lbI, lI));
    const int rImin = min(raI, min(rbI, rI));
    const int rImax = max(raI, max(rbI, rI));

    *sum0 += min(max(0, max(lI - rImax, rImin - lI)),
              max(0, max(rI - lImax, lImin - rI)));


}

__device__ void costAtDispImg_v3_1( int* sum1, int x, int y, int d)
{
    // // computaion on the Sobel domain
    const int lI1 = tex2D(ltex1, x, y);
    const int rI1 = tex2D(rtex1, x - d, y);
    //NOTE: d >= 0 !!!
    const int laI1 = (lI1 + tex2D(ltex1, x-1, y))/2;  //x > 0 ? (lI + tex2D(ltex, x - 1, y))/2 : lI;
    const int lbI1 = (lI1 + tex2D(ltex1, x+1, y))/2; //x < width-1 ? (lI + tex2D(ltex, x + 1, y))/2 : lI;
    const int raI1 = (rI1 + tex2D(rtex1, x-d-1, y))/2;  //x > d ? 0.5f*(rI + tex2D(rtex, x - d - 1, y)) : rI;
    const int rbI1 = (rI1 + tex2D(rtex1, x-d+1, y))/2; //x < width+d-1 ?  (rI + tex2D(rtex, x - d + 1, y))/2 : rI;
    const int lImin1 = min(laI1, min(lbI1, lI1));
    const int lImax1 = max(laI1, max(lbI1, lI1));
    const int rImin1 = min(raI1, min(rbI1, rI1));
    const int rImax1 = max(raI1, max(rbI1, rI1));

    *sum1 += min(max(0, max(lI1 - rImax1, rImin1 - lI1)),
               max(0, max(rI1 - lImax1, lImin1 - rI1)));

}

// Macro definition
// #define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d! \n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

__global__ void cost_Pix_BT_img_v3(short *d_CPixbuf, int width, int height,int img_width, int minD,
                                    int maxD, float sobel_multiply_scale, float img_multiply_scale,
                                    int scale_h, int scale_w, int step_mode){

    //int D = maxD - minD;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x < width && y < height ){

        for (int d=minD; d < maxD; d++)
        {
            short *costptr = d_CPixbuf + (d-minD)*width*height + y*width + x;

            int cost_img = 0; int cost_Sobel = 0;

            // CudaAssert( x-d < width );



            if(x*scale_w - d <= 1 ){
            //if(x < maxD){
                costptr[0] = 100; //(short)(costptr[d-minD] + (cost >> diff_scale));
            }else{
                
                if(step_mode == 0) {
                    costAtDispImg_v3_0( &cost_img, x * scale_w, y * scale_h, d);
                    costAtDispImg_v3_1( &cost_Sobel, x * scale_w, y * scale_h, d);

                } else {
                    for(int j = 0; j < scale_w; j ++) {
                        costAtDispImg_v3_0( &cost_img, x * scale_w + j, y * scale_h, d);
                        costAtDispImg_v3_1( &cost_Sobel, x * scale_w + j, y * scale_h, d);
                    }


                }

                costptr[0] = (short) ( cost_img * img_multiply_scale + cost_Sobel * sobel_multiply_scale);


            }
        }
    }
}

//box filter -- sum
__global__ void cost_Block_BT(short* d_out, const int width, const int height, const int SW, const int SH, const int P2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure the current thread is inside the image bounds
    ////NOTE: height-SH just for the same comparison with OPENCV lib function
    //if( x < width && y < height-SH)
    if( x < width && y < height)
    {
        short output_value0 = 0.0f;
        short output_value1 = 0.0f;
        short output_value2 = 0.0f;
        short output_value3 = 0.0f;
        short* outptr0 = d_out + y*width + x;
        short* outptr1 = d_out + width*height + y*width + x;
        short* outptr2 = d_out + 2*width*height + y*width + x;
        short* outptr3 = d_out + 3*width*height + y*width + x;
        //Sum the window pixels
        for(int j= - SH; j <= SH; j++)
        {
            for(int i = - SW; i <= SW; i++)
            {
                //No need to worry about Out-Of-Range access. tex2D automatically handles it.
                output_value0 += tex2D(short_tex0, x+i, y+j);
                output_value1 += tex2D(short_tex1, x+i, y+j);
                output_value2 += tex2D(short_tex2, x+i, y+j);
                output_value3 += tex2D(short_tex3, x+i, y+j);
            }
        }
        //outptr0[0] = static_cast<short>(output_value0);
        outptr0[0] =  output_value0; // + P2;
        outptr1[0] =  output_value1; // + P2;
        outptr2[0] =  output_value2; // + P2;
        outptr3[0] =  output_value3; // + P2;

    }
}
#ifndef KERNEL_TEMPLATE
#define KERNEL_TEMPLATE
#define calcSstep1_template(disp) if(disp_num == disp) { calcSstep1<disp><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1, d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);}
#define calcSstep2_top2bottom_template(disp) if(disp_num == disp) { calcSstep2_top2bottom<disp,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST); calcSstep2_bottom2top<disp,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);}
#define calcSstep2_left2right_template(disp) if(disp_num == disp) { calcSstep2_left2right<disp,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST); calcSstep2_right2left<disp,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);}
#define calc_Disp_template(disp) if(disp_num == disp) { calc_Disp<disp, CostType, DispType, PixType><<<Dispgrid1, Dispblock1>>>(d_CCSbuf, d_disp1, d_disp1cost_tmp, d_disp1_tmp, d_udispconf, d_disp2cost, d_disp2_tran, width1, height, minD, MAX_COST, confRatio, uniquenessRatio, INVALID_DISP_SCALED);}
#endif
__host__ void computeSGBM(const short*cost_volume,
        int img_width, int img_height, short *disp1, unsigned char *udispconf,
        int disp_width, int disp_height, int min_disp, int disp_num){

    const CostType MAX_COST = SHRT_MAX;

    int minD = min_disp;
    int maxD = minD + disp_num; //64; //atio(argv[2]);
    int SADWindowSize_width = 3;
    int SADWindowSize_height = 3;
    int uniquenessRatio = 10;
    int confRatio = 50;
    int sdisp2MaxDiff = 1;
    int P1 = 8*1*SADWindowSize_width*SADWindowSize_height, P2 = 4*P1;
    //int k;
    int width = disp_width, height = disp_height;
    int minX1 = 0, maxX1 = width;
    int D = maxD - minD, width1 = maxX1 - minX1;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;

    size_t costBufSize = width1 * D;
    size_t CSBufSize = costBufSize*height;


    CostType *d_CCSbuf, *d_Sbuf, *d_Lrbuf;
    cudaMalloc((void**) &d_CCSbuf, CSBufSize*sizeof(CostType));
    cudaMalloc((void**) &d_Sbuf, CSBufSize*sizeof(CostType));
    cudaMalloc((void**) &d_Lrbuf, ((width1+2)*2+D*(width1+2)*2)*3*sizeof(CostType));
    cudaMemset(d_Lrbuf, 0, ((width1+2)*2+D*(width1+2)*2)*3*sizeof(CostType));

    cudaMemset(d_CCSbuf, 0, CSBufSize*sizeof(CostType));
    cudaMemset(d_Sbuf, 0, CSBufSize*sizeof(CostType));

    cudaStream_t stream1;
    cudaError_t result = cudaStreamCreate(&stream1);
    transpose<CostType>(0, stream1, D, height*width1, cost_volume, d_CCSbuf);
    //result = cudaStreamDestroy(stream1);

    /***********************calculate S*********************************/

    int batch = 1;
    int numblocks = ceil((float) width1/batch ); // single pass = 4 single directions per time
    dim3 numthreads = D; // one thread calculate one disparity value

    CostType *d_Lr0, *d_Lr1, *d_minLr0, *d_minLr1, *d_swap;
    d_Lr0 = d_Lrbuf; d_Lr1 = d_Lrbuf + (width1+2)*3*D;
    d_minLr0 = d_Lrbuf + (width1+2)*3*D*2;
    d_minLr1 = d_Lrbuf + (width1+2)*3*D*2 + (width1+2)*3;

#pragma unroll
    for(int y= 0; y < height; y++){
        calcSstep1_template(16)
        else calcSstep1_template(32)
        else calcSstep1_template(48)
        else calcSstep1_template(64)
        else calcSstep1_template(80)
        else calcSstep1_template(96)
        else calcSstep1_template(112)
        else calcSstep1_template(128)
        else calcSstep1_template(144)
        else calcSstep1_template(160)
        else calcSstep1_template(176)
        else calcSstep1_template(192)
        else calcSstep1_template(208)
        else calcSstep1_template(224)
        else calcSstep1_template(240)
        else calcSstep1_template(256)


        // if(disp_num == 64) {
        //     calcSstep1<64><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 48) {
        //     calcSstep1<48><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 32) {
        //     calcSstep1<32><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 16) {
        //     calcSstep1<16><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // }

        d_swap = d_Lr1; d_Lr1 = d_Lr0; d_Lr0 = d_swap;
        d_swap = d_minLr1; d_minLr1 = d_minLr0; d_minLr0 = d_swap;
    }


    cudaMemset(d_Lrbuf, 0, ((width1+2)*2+D*(width1+2)*2)*3*sizeof(CostType));
#pragma unroll
    for(int y = height-1; y > -1; y--){
        calcSstep1_template(16)
        else calcSstep1_template(32)
        else calcSstep1_template(48)
        else calcSstep1_template(64)
        else calcSstep1_template(80)
        else calcSstep1_template(96)
        else calcSstep1_template(112)
        else calcSstep1_template(128)
        else calcSstep1_template(144)
        else calcSstep1_template(160)
        else calcSstep1_template(176)
        else calcSstep1_template(192)
        else calcSstep1_template(208)
        else calcSstep1_template(224)
        else calcSstep1_template(240)
        else calcSstep1_template(256)

        // if(disp_num == 64) {
        //     calcSstep1<64><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 48) {
        //     calcSstep1<48><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 32) {
        //     calcSstep1<32><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // } else if (disp_num == 16) {
        //     calcSstep1<16><<<numblocks, numthreads>>>(d_Sbuf, d_CCSbuf, d_Lr0, d_Lr1,
        //         d_minLr0, d_minLr1, y, width1, D, batch, P1, P2);
        // }

        d_swap = d_Lr1; d_Lr1 = d_Lr0; d_Lr0 = d_swap;
        d_swap = d_minLr1; d_minLr1 = d_minLr0; d_minLr0 = d_swap;
    }

    int numblocks0 = ceil((float)width/2.0);//height;
    dim3 blocksize0(D, 2); //int numthreads1 = D;

    calcSstep2_top2bottom_template(16)
    else calcSstep2_top2bottom_template(32)
    else calcSstep2_top2bottom_template(48)
    else calcSstep2_top2bottom_template(64)
    else calcSstep2_top2bottom_template(80)
    else calcSstep2_top2bottom_template(96)
    else calcSstep2_top2bottom_template(112)
    else calcSstep2_top2bottom_template(128)
    else calcSstep2_top2bottom_template(144)
    else calcSstep2_top2bottom_template(160)
    else calcSstep2_top2bottom_template(176)
    else calcSstep2_top2bottom_template(192)
    else calcSstep2_top2bottom_template(208)
    else calcSstep2_top2bottom_template(224)
    else calcSstep2_top2bottom_template(240)
    else calcSstep2_top2bottom_template(256)

    // if(disp_num == 64) {
    //     calcSstep2_top2bottom<64,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_bottom2top<64,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 48) {
    //     calcSstep2_top2bottom<48,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_bottom2top<48,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 32) {
    //     calcSstep2_top2bottom<32,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_bottom2top<32,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 16) {
    //     calcSstep2_top2bottom<16,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_bottom2top<16,2><<<numblocks0, blocksize0>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // }


    int numblocks1 = ceil((float)height/2.0);//height;
    dim3 blocksize1(D, 2); //int numthreads1 = D;

    // if(disp_num == 64) {
    //     calcSstep2_left2right<64,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_right2left<64,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 48) {
    //     calcSstep2_left2right<48,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_right2left<48,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 32) {
    //     calcSstep2_left2right<32,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_right2left<32,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // } else if (disp_num == 16) {
    //     calcSstep2_left2right<16,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    //     calcSstep2_right2left<16,2><<<numblocks1, blocksize1>>>(d_Sbuf, d_CCSbuf, width1, height, D, P1, P2, MAX_COST);
    // }
    calcSstep2_left2right_template(16)
    else calcSstep2_left2right_template(32)
    else calcSstep2_left2right_template(48)
    else calcSstep2_left2right_template(64)
    else calcSstep2_left2right_template(80)
    else calcSstep2_left2right_template(96)
    else calcSstep2_left2right_template(112)
    else calcSstep2_left2right_template(128)
    else calcSstep2_left2right_template(144)
    else calcSstep2_left2right_template(160)
    else calcSstep2_left2right_template(176)
    else calcSstep2_left2right_template(192)
    else calcSstep2_left2right_template(208)
    else calcSstep2_left2right_template(224)
    else calcSstep2_left2right_template(240)
    else calcSstep2_left2right_template(256)



    /************************calculate disparity map****************************/

    transpose<CostType>(0, stream1, height*width, D, d_Sbuf, d_CCSbuf);

    DispType *d_disp1;
    PixType *d_disp1_tmp, *d_udispconf; // *d_disp2, *d_disp2cost; //*d_Lrbuf;
    CostType *d_disp2cost, *d_disp1cost_tmp;
    PixType *d_disp1_tmp_tran;
    CostType *d_disp1cost_tmp_tran;
    DispType *d_disp2, *d_disp2_tran;

    cudaMalloc((void**) &d_disp1, width1*height*sizeof(DispType));
    cudaMalloc((void**) &d_disp1_tmp, width1*height*sizeof(PixType));
    cudaMalloc((void**) &d_disp1_tmp_tran, width1*height*sizeof(PixType));
    cudaMalloc((void**) &d_udispconf, width1*height*sizeof(PixType));
    cudaMalloc((void**) &d_disp2, width1*height*sizeof(DispType));
    cudaMalloc((void**) &d_disp2_tran, width1*height*sizeof(DispType));
    cudaMalloc((void**) &d_disp2cost, width1*height*sizeof(CostType));
    cudaMalloc((void**) &d_disp1cost_tmp, width1*height*sizeof(CostType));
    cudaMalloc((void**) &d_disp1cost_tmp_tran, width1*height*sizeof(CostType));

    cudaMemset(d_udispconf, MY_INVALID_DISP, width1*height*sizeof(PixType));
    //cudaMemset(d_disp2_tran, INVALID_DISP_SCALED, width1*height*sizeof(PixType));

    dim3 Dispblock1(16, 16);
    dim3 Dispgrid1(divUp(width1, 16), divUp(height, 16)); //int numthreads1 = D;


    calc_Disp_template(16)
    else calc_Disp_template(32)
    else calc_Disp_template(48)
    else calc_Disp_template(64)
    else calc_Disp_template(80)
    else calc_Disp_template(96)
    else calc_Disp_template(112)
    else calc_Disp_template(128)
    else calc_Disp_template(144)
    else calc_Disp_template(160)
    else calc_Disp_template(176)
    else calc_Disp_template(192)
    else calc_Disp_template(208)
    else calc_Disp_template(224)
    else calc_Disp_template(240)
    else calc_Disp_template(256)


    // if(disp_num == 64) {
    //     calc_Disp<64, CostType, DispType, PixType><<<Dispgrid1,
    //     Dispblock1>>>(d_CCSbuf, d_disp1, d_disp1cost_tmp, d_disp1_tmp,
    //             d_udispconf, d_disp2cost, d_disp2_tran, width1, height, minD, MAX_COST,
    //         confRatio, uniquenessRatio, INVALID_DISP_SCALED);
    // } else if(disp_num == 32) {
    //     calc_Disp<32, CostType, DispType, PixType><<<Dispgrid1,
    //     Dispblock1>>>(d_CCSbuf, d_disp1, d_disp1cost_tmp, d_disp1_tmp,
    //             d_udispconf, d_disp2cost, d_disp2_tran, width1, height, minD, MAX_COST,
    //         confRatio, uniquenessRatio, INVALID_DISP_SCALED);

    // } else if(disp_num == 48) {
    //     calc_Disp<48, CostType, DispType, PixType><<<Dispgrid1,
    //     Dispblock1>>>(d_CCSbuf, d_disp1, d_disp1cost_tmp, d_disp1_tmp,
    //             d_udispconf, d_disp2cost, d_disp2_tran, width1, height, minD, MAX_COST,
    //         confRatio, uniquenessRatio, INVALID_DISP_SCALED);

    // } else if(disp_num == 16) {
    //     calc_Disp<16, CostType, DispType, PixType><<<Dispgrid1,
    //     Dispblock1>>>(d_CCSbuf, d_disp1, d_disp1cost_tmp, d_disp1_tmp,
    //             d_udispconf, d_disp2cost, d_disp2_tran, width1, height, minD, MAX_COST,
    //         confRatio, uniquenessRatio, INVALID_DISP_SCALED);

    // }


    //cudaStream_t stream2;
    //result = cudaStreamCreate(&stream2);
    transpose<PixType>(0, stream1, height, width, d_disp1_tmp, d_disp1_tmp_tran);
    transpose<CostType>(0, stream1, height, width, d_disp1cost_tmp, d_disp1cost_tmp_tran);

    pre_consistance_check_tran<PixType, CostType, DispType><<<divUp(height,16), 16>>>(d_disp1_tmp_tran, d_disp1cost_tmp_tran, d_disp2cost, d_disp2_tran, width, height, minX1, minD);


    transpose<DispType>(0, stream1, width, height, d_disp2_tran, d_disp2);
    result = cudaStreamDestroy(stream1);

    dim3 Dispblock2(16, 16);
    dim3 Dispgrid2(divUp(width1, 16), divUp(height, 16)); //int numthreads1 = D;

    consistance_check<DispType, PixType><<<Dispgrid2, Dispblock2>>>(d_disp2,
            d_disp1, d_udispconf, width, height, INVALID_DISP_SCALED, minD,
            sdisp2MaxDiff);

   /*********************END*****************************/
    cudaMemcpy(disp1, d_disp1, width*height*sizeof(DispType), cudaMemcpyDeviceToHost);
    cudaMemcpy(udispconf, d_udispconf, width*height*sizeof(PixType), cudaMemcpyDeviceToHost);

    //   cudaUnbindTexture(ltex); cudaUnbindTexture(rtex);
//    cudaFreeArray(d_left); cudaFreeArray(d_right);
    cudaFree(d_Sbuf); cudaFree(d_Lrbuf);
    cudaFree(d_CCSbuf); cudaFree(d_disp1);
    cudaFree(d_disp1_tmp); cudaFree(d_udispconf); cudaFree(d_disp2);
    cudaFree(d_disp2cost); cudaFree(d_disp1cost_tmp);
    cudaFree(d_disp1_tmp_tran); cudaFree(d_disp1cost_tmp_tran);
    cudaFree(d_disp2_tran);

    return;
}


void disp2color(const float *fdisp, unsigned char *color, int width, int height, int minDisparity, int numDisparities) {

    //int r,g,b;
    float value;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            value = fdisp[i*width + j]/(minDisparity+numDisparities-1);
            if (value>=0&&value<=0.11)
            {
                color[i*width*3+j*3+0] = value/0.11*112+143;
                color[i*width*3+j*3+1] = 0;
                color[i*width*3+j*3+2] = 0;
            }
            else if (value>0.11&&value<=0.125)
            {
                color[i*width*3+j*3+0] = 255;
                //color[i*width*3+j*3+1] = 0;
                //color[i*width*3+j*3+2] = 0;
            }
            else if (value>0.125&&value<=0.36)
            {
                color[i*width*3+j*3+0] = 255;
                color[i*width*3+j*3+1] = (value-0.125)/0.235*255;
                color[i*width*3+j*3+2] = 0;
            }
            else if (value>0.36&&value<=0.375)
            {
                color[i*width*3+j*3+0] = 255;
                color[i*width*3+j*3+1] = 255;
                color[i*width*3+j*3+2] = 0;
            }
            else if (value>0.375&&value<=0.61)
            {
                color[i*width*3+j*3+0] = 255-(value-0.375)/0.235*255;
                color[i*width*3+j*3+1] = 255;
                color[i*width*3+j*3+2] = (value-0.375)/0.235*255;
            }
            else if (value>0.61&&value<=0.625)
            {
                color[i*width*3+j*3+0] = 0;
                color[i*width*3+j*3+1] = 255;
                color[i*width*3+j*3+2] = 255;
            }
            else if (value>0.625&&value<=0.86)
            {
                color[i*width*3+j*3+0] = 0;
                color[i*width*3+j*3+1] = 255-(value-0.625)/0.235*255;
                color[i*width*3+j*3+2] = 255;
            }
            else if (value>0.86&&value<=0.875)
            {
                color[i*width*3+j*3+0] = 0;
                color[i*width*3+j*3+1] = 0;
                color[i*width*3+j*3+2] = 255;
            }
            else if (value>0.875&&value<1)
            {
                color[i*width*3+j*3+0] = 0;
                color[i*width*3+j*3+1] = 0;
                color[i*width*3+j*3+2] = 255-(value-0.875)/0.125*127;
            }
            else
            {
                color[i*width*3+j*3+0] = 0;
                color[i*width*3+j*3+1] = 0;
                color[i*width*3+j*3+2] = 0;
            }

        }
}

} // namespace caffe
