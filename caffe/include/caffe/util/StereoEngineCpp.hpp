#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>

#include "time.h"
#include <sys/time.h>
#ifdef USE_NEON
#include <arm_neon.h>
#endif

#include <vector>
#include <iostream>
#include <iterator>
#include <cstdio>

using namespace cv;

#define DISP_SCALE 16
#define DISP_SHIFT  4
#define APPROXIMATE_FLOAT_ZERO (1e-43f)

typedef uchar PixType;
typedef short CostType;
typedef short DispType;
typedef cv::Point_<short> Point2s;

namespace caffe {

    static void calcPixelCostBT( const Mat& img1, const Mat& img2, const int disp_width, int y, int minD, int maxD, CostType* cost,
                                PixType* buffer, const PixType* tab, int tabOfs, int, int step,
                                float sobel_multiply_scale, float img_multiply_scale, int step_mode ) {
              int x, c, width = img1.cols, cn = img1.channels();
              int minX1 = 0, maxX1 = width;
              int minX2 = 0, maxX2 = width;
              int D = maxD - minD, width2 = maxX2 - minX2; //, width1 = maxX1 - minX1;
              const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
              PixType *prow1 = buffer + width2*2, *prow2 = prow1 + width*cn*2;

              tab += tabOfs;
              for( c = 0; c < cn*2; c++ ) {

                  prow1[width*c] = prow1[width*c + width-1] =
                      prow2[width*c] = prow2[width*c + width-1] = tab[0];

              }

              int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows-1 ? (int)img1.step : 0;
              int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows-1 ? (int)img2.step : 0;

              if( cn == 1 ) {
                  for( x = 1; x < width-1; x++ ) {
                      prow1[x] = tab[(row1[x+1] - row1[x-1])*2 + row1[x+n1+1] - row1[x+n1-1] + row1[x+s1+1] - row1[x+s1-1]];
                      prow2[width-1-x] = tab[(row2[x+1] - row2[x-1])*2 + row2[x+n2+1] - row2[x+n2-1] + row2[x+s2+1] - row2[x+s2-1]];

                      prow1[x+width] = row1[x];
                      prow2[width-1-x+width] = row2[x];
                  }
              } else {
                  for( x = 1; x < width-1; x++ ) {
                      prow1[x] = tab[(row1[x*3+3] - row1[x*3-3])*2 + row1[x*3+n1+3] - row1[x*3+n1-3] + row1[x*3+s1+3] - row1[x*3+s1-3]];
                      prow1[x+width] = tab[(row1[x*3+4] - row1[x*3-2])*2 + row1[x*3+n1+4] - row1[x*3+n1-2] + row1[x*3+s1+4] - row1[x*3+s1-2]];
                      prow1[x+width*2] = tab[(row1[x*3+5] - row1[x*3-1])*2 + row1[x*3+n1+5] - row1[x*3+n1-1] + row1[x*3+s1+5] - row1[x*3+s1-1]];

                      prow2[width-1-x] = tab[(row2[x*3+3] - row2[x*3-3])*2 + row2[x*3+n2+3] - row2[x*3+n2-3] + row2[x*3+s2+3] - row2[x*3+s2-3]];
                      prow2[width-1-x+width] = tab[(row2[x*3+4] - row2[x*3-2])*2 + row2[x*3+n2+4] - row2[x*3+n2-2] + row2[x*3+s2+4] - row2[x*3+s2-2]];
                      prow2[width-1-x+width*2] = tab[(row2[x*3+5] - row2[x*3-1])*2 + row2[x*3+n2+5] - row2[x*3+n2-1] + row2[x*3+s2+5] - row2[x*3+s2-1]];

                      prow1[x+width*3] = row1[x*3];
                      prow1[x+width*4] = row1[x*3+1];
                      prow1[x+width*5] = row1[x*3+2];

                      prow2[width-1-x+width*3] = row2[x*3];
                      prow2[width-1-x+width*4] = row2[x*3+1];
                      prow2[width-1-x+width*5] = row2[x*3+2];
                  }
              }
              // memset( cost, 0, width1*D*sizeof(cost[0]) );
              memset( cost, 0, disp_width*D*sizeof(cost[0]) );

              buffer -= minX2;
              cost -= minX1*D + minD; // simplify the cost indices inside the loop

              int stride = 1;
              if (step_mode == 0){
                stride = step;
              }

              for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width ) {
                  // int diff_scale = c < cn ? 0 : 2;
                  float diff_scale = c < cn ? sobel_multiply_scale : img_multiply_scale;
                  // precompute
                  //   v0 = min(row2[x-1/2], row2[x], row2[x+1/2]) and
                  //   v1 = max(row2[x-1/2], row2[x], row2[x+1/2]) and
                  for( x = minX2; x < maxX2; x++ ) {
                      int v = prow2[x];
                      int vl = x > 0 ? (v + prow2[x-1])/2 : v;
                      int vr = x < width-1 ? (v + prow2[x+1])/2 : v;
                      int v0 = min(vl, vr); v0 = min(v0, v);
                      int v1 = max(vl, vr); v1 = max(v1, v);
                      buffer[x] = (PixType)v0;
                      buffer[x + width2] = (PixType)v1;
                  }

                  for( x = minX1; x < maxX1; x+=stride ) {
                      int u = prow1[x];
                      int ul = x > 0 ? (u + prow1[x-1])/2 : u;
                      int ur = x < width-1 ? (u + prow1[x+1])/2 : u;
                      int u0 = min(ul, ur); u0 = min(u0, u);
                      int u1 = max(ul, ur); u1 = max(u1, u);

                      {
                          for( int d = minD; d < maxD; d++ ) {


                              int v = prow2[width-x-1 + d];
                              int v0 = buffer[width-x-1 + d];
                              int v1 = buffer[width-x-1 + d + width2];
                              int c0 = max(0, u - v1); c0 = max(c0, v0 - u);
                              int c1 = max(0, v - u1); c1 = max(c1, u0 - v);

                              cost[x/step*D + d] = (CostType)(cost[x/step*D+d] + (min(c0, c1) * diff_scale));
                          }
                      }


                  }
              }

          }

    static void InitBuffer(const Mat& img1, Mat& buffer, int minD, int nun_disparity, int disp_height, int disp_width){
        int maxD = minD + nun_disparity;
        Size SADWindowSize = Size(3, 3);
        int  SH2 = SADWindowSize.height/2;

        const bool fullDP = false; //params.fullDP_;

        int NR = 16; //params.NR_;
        int NR2 = NR/2;
        int width = disp_width, height = disp_height;
        int minX1 = 0, maxX1 = width;
        int D = maxD - minD, width1 = maxX1 - minX1;
        int D2 = D+16;

        const int NLR = 2; //params.NLR_;
        const int LrBorder = NLR - 1;

        size_t costBufSize = width1*D;
        size_t CBufSize = costBufSize*height;
        size_t SBufSize = costBufSize*(fullDP ? height : 1);
        size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
        int hsumBufNRows = SH2*2 + 2;
        int img_width = img1.cols;
        size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
            costBufSize*(hsumBufNRows + 1)*sizeof(CostType) + // hsumBuf, pixdiff
            (CBufSize + SBufSize)*sizeof(CostType) + // C, S
            img_width*16*img1.channels()*sizeof(PixType) + // temp buffer for computing per-pixel cost
            img_width*(sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2

        if( !buffer.data || !buffer.isContinuous() || buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
            buffer.create(1, (int)totalBufSize, CV_8U);
    }


    static void ComputeBTCostVolume( const Mat& img1, const Mat& img2, Mat& buffer, int minD,
                                        int nun_disparity, int step_h, int step_w,
                                        int disp_height, int disp_width,
                                        float sobel_multiply_scale, float img_multiply_scale, int step_mode ) {


        bool fullDP = false; //params.fullDP_;
        int NR = 16; //params.NR_,
        int NR2 = NR/2;
        const int ALIGN = 16; //params.ALIGN_;
        int ftzero = 63;

        int maxD = minD + nun_disparity; //params.getNumberOfDisparities();
        Size SADWindowSize = Size(3, 3); //params.SADWindowSize_;

        // int P1 = 8*img1.channels()*SADWindowSize.width*SADWindowSize.height, P2 = 4*P1;
        int k, width = disp_width, height = disp_height;
        int minX1 = 0, maxX1 = width;
        int D = maxD - minD, width1 = maxX1 - minX1;
        int SW2 = SADWindowSize.width/2, SH2 = SADWindowSize.height/2;
        const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
        PixType clipTab[TAB_SIZE];

        for( k = 0; k < TAB_SIZE; k++ )
            clipTab[k] = (PixType)(min(max(k - TAB_OFS, -ftzero), ftzero) + ftzero);


        CV_Assert( D % 16 == 0 );

        // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
        // if you change NR, please, modify the loop as well.
        int D2 = D+16;
        // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
        // for 8-way dynamic programming we need the current row and
        // the previous row, i.e. 2 rows in total
        const int NLR = 2; //params.NLR_;
        const int LrBorder = NLR - 1;


        // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
        // we keep pixel difference cost (C) and the summary cost over NR directions (S).
        // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
        size_t costBufSize = width1*D;
        size_t CBufSize = costBufSize*height;
        size_t SBufSize = costBufSize*(fullDP ? height : 1);
        size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
        int hsumBufNRows = SH2*2 + 2;

        // summary cost over different (nDirs) directions
        CostType* Cbuf = (CostType*)alignPtr(buffer.data, ALIGN);
        CostType* Sbuf = Cbuf + CBufSize;
        CostType* hsumBuf = Sbuf + SBufSize;
        CostType* pixDiff = hsumBuf + costBufSize*hsumBufNRows;

        CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
        DispType* disp2ptr = (DispType*)(disp2cost + width);
        PixType* tempBuf = (PixType*)(disp2ptr + width);

        // add P2 to every C(x,y). it saves a few operations in the inner loops
        //for( k = 0; k < width1*D; k++ )
        //    Cbuf[k] = (CostType)P2;
        for( k = 0; k < width1*D; k++ )
            Cbuf[k] = (CostType)0;

        int  y1, y2,  dy;
        y1 = 0; y2 = height; dy = 1;

        for( int y = y1; y != y2; y += dy ) {
            int x, d;
            CostType* C = Cbuf +  y*costBufSize;

            int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

            for( k = dy1; k <= dy2; k++ ) {
                CostType* hsumAdd = hsumBuf + (min(k, height-1) % hsumBufNRows)*costBufSize;

                if( k < height ) {

                    calcPixelCostBT( img1, img2, disp_width ,  k*step_h, minD, maxD, pixDiff,
                                      tempBuf, clipTab, TAB_OFS, ftzero, step_w,
                                      sobel_multiply_scale, img_multiply_scale, step_mode);

                    memset(hsumAdd, 0, D*sizeof(CostType));
                    for( x = 0; x <= SW2*D; x += D ) {
                        int scale = x == 0 ? SW2 + 1 : 1;
                        for( d = 0; d < D; d++ )
                            hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
                    }

                    if( y > 0 ) {
                        const CostType* hsumSub = hsumBuf + (max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                        const CostType* Cprev =  C - costBufSize;

                        for( x = D; x < width1*D; x += D ) {
                            const CostType* pixAdd = pixDiff + min(x + SW2*D, (width1-1)*D);
                            const CostType* pixSub = pixDiff + max(x - (SW2+1)*D, 0);

#ifdef USE_NEON
                            {
                                for( d = 0; d < D; d += 16 ) {
                                    int16x8_t v_hv0 = vld1q_s16(hsumAdd + x - D + d);
                                    int16x8_t v_hv1 = vld1q_s16(hsumAdd + x - D + d +8);
                                    int16x8_t v_pA0 = vld1q_s16(pixAdd + d);
                                    int16x8_t v_pA1 = vld1q_s16(pixAdd + d + 8);
                                    int16x8_t v_pS0 = vld1q_s16(pixSub + d);
                                    int16x8_t v_pS1 = vld1q_s16(pixSub + d + 8);
                                    int16x8_t v_Cx0 = vld1q_s16(Cprev + x + d);
                                    int16x8_t v_Cx1 = vld1q_s16(Cprev + x + d + 8);
                                    int16x8_t v_hsb0 = vld1q_s16(hsumSub + x + d);
                                    int16x8_t v_hsb1 = vld1q_s16(hsumSub + x + d + 8);

                                    v_hv0 = vaddq_s16(v_hv0, vsubq_s16(v_pA0, v_pS0));
                                    v_hv1 = vaddq_s16(v_hv1, vsubq_s16(v_pA1, v_pS1));
                                    v_Cx0 = vaddq_s16(v_Cx0, vsubq_s16(v_hv0, v_hsb0));
                                    v_Cx1 = vaddq_s16(v_Cx1, vsubq_s16(v_hv1, v_hsb1));

                                    vst1q_s16(hsumAdd + x + d, v_hv0);
                                    vst1q_s16(hsumAdd + x + d + 8, v_hv1);
                                    vst1q_s16(C + x + d, v_Cx0);
                                    vst1q_s16(C + x + d + 8, v_Cx1);
                                }
                            }
#else

                            {
                                for( d = 0; d < D; d++ ) {
                                    int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                    C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);

                                }
                            }
#endif //USE_NEON

                        }
                    } else {
                        for( x = D; x < width1*D; x += D ) {
                            const CostType* pixAdd = pixDiff + min(x + SW2*D, (width1-1)*D);
                            const CostType* pixSub = pixDiff + max(x - (SW2+1)*D, 0);
#ifdef USE_NEON
                            for(d = 0; d < D; d += 16) {
                                int16x8_t v_hsa0 = vld1q_s16(hsumAdd + x - D + d);
                                int16x8_t v_hsa1 = vld1q_s16(hsumAdd + x - D + d + 8);
                                int16x8_t v_pA0 = vld1q_s16(pixAdd + d);
                                int16x8_t v_pA1 = vld1q_s16(pixAdd + d + 8);
                                int16x8_t v_pSb0 = vld1q_s16(pixSub + d);
                                int16x8_t v_pSb1 = vld1q_s16(pixSub + d + 8);

                                v_hsa0 = vaddq_s16(v_hsa0, vsubq_s16(v_pA0, v_pSb0));
                                v_hsa1 = vaddq_s16(v_hsa1, vsubq_s16(v_pA1, v_pSb1));

                                vst1q_s16(hsumAdd + d, v_hsa0);
                                vst1q_s16(hsumAdd + d + 8, v_hsa1);
                            }
#else
                            for( d = 0; d < D; d++ )
                                hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
#endif //USE_NEON
                        }
                    }
                }

                if( y == 0 ) {
                    int scale = k == 0 ? SH2 + 1 : 1;
#ifdef USE_NEON
                    int16x8_t v_scale = vdupq_n_s16((short) scale);
                    for(x = 0; x < width1*D; x += 16) {
                        int16x8_t v_C0 = vld1q_s16(C + x);
                        int16x8_t v_C1 = vld1q_s16(C + x + 8);
                        //int16x8_t v_C2 = vld1q_s16(C + x + 16);
                        //int16x8_t v_C3 = vld1q_s16(C + x + 24);

                        int16x8_t v_hA0 = vld1q_s16(hsumAdd + x);
                        int16x8_t v_hA1 = vld1q_s16(hsumAdd + x + 8);
                        //int16x8_t v_hA2 = vld1q_s16(hsumAdd + x + 16);
                        //int16x8_t v_hA3 = vld1q_s16(hsumAdd + x + 24);

                        v_C0 = vmlaq_s16(v_C0, v_hA0, v_scale);
                        v_C1 = vmlaq_s16(v_C1, v_hA1, v_scale);
                        //v_C2 = vmlaq_s16(v_C2, v_hA2, v_scale);
                        //v_C3 = vmlaq_s16(v_C3, v_hA3, v_scale);

                        vst1q_s16(C + x, v_C0);
                        vst1q_s16(C + x + 8, v_C1);
                        //vst1q_s16(C + x + 16, v_C2);
                        //vst1q_s16(C + x + 24, v_C3);
                    }

#else
                    for( x = 0; x < width1*D; x++ )
                        C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
#endif
                }
            }
        }
    }
};
