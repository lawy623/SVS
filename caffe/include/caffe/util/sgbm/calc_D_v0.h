#ifndef CALC_D_V0_H__
#define CALC_D_V0_H__



#ifndef DISP_CONSTANT
#define DISP_CONSTANT
#define DISP_SCALE (16)
#define DISP_SHIFT (4)
enum{MY_INVALID_DISP=0, MY_MED_DISP=128, MY_CONF_DISP=255};
#endif


template <int D, typename T_src, typename T_dst, typename T_dst2>
__global__ void calc_Disp(T_src* d_S, T_dst* d_disp1, T_src* d_disp1cost_tmp, T_dst2* d_disp1_tmp, T_dst2* d_udispconf, T_src* d_disp2cost, T_dst* d_disp2_tran, int width, int height, int minD, T_src MAX_COST, int confRatio, int uniquenessRatio, int INVALID_DISP_SCALED) {
    T_src Sbuf[D];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
/*
    if(x < width && y < height){
        disp2cost[x] = MAX_COST; 
        disp2ptr[x] = (T_src) INVALID_DISP_SCALED;
    }
*/
    
    if(x < width && y < height) {

        //int d, minS = MAX_COST, bestDisp = -1;
        T_src d, minS = MAX_COST, bestDisp = -1;
        T_dst* sdispptr = d_disp1 + y*width + x;
        T_dst2* dispptr_tmp = d_disp1_tmp + y*width + x;
        T_dst2* confptr = d_udispconf + y*width + x;
        T_src* cost1tmpptr = d_disp1cost_tmp + y*width + x;
        
        *(d_disp2cost + y*width + x) = MAX_COST;
        *(d_disp2_tran + y*width +x) = INVALID_DISP_SCALED; 

        for(d = 0; d < D; d++) {
            T_src* Sptr = d_S + d*width*height + y*width + x;
            T_src Sval =  Sptr[0];
            Sbuf[d] = Sval;
            if( Sval < minS ){
                minS = Sval;
                bestDisp = d;
            }
        }
        
        // save the bestDisp for left-right consistance check    
        dispptr_tmp[0] = (T_dst2) bestDisp;
        cost1tmpptr[0] = minS;
        d = bestDisp;

        //do subpixel quadratic interpolation
        // fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
        // then find minimum of the parabola.
        if( 0 < d && d < D-1) {
            T_src denom2 = max(Sbuf[d-1] + Sbuf[d+1] - 2*Sbuf[d], 1);
            d = d*DISP_SCALE + ((Sbuf[d-1] - Sbuf[d+1])*DISP_SCALE + denom2)/(denom2*2);
        } else {
            d *= DISP_SCALE;
        }
        
        sdispptr[0] = (T_dst) (d + minD*DISP_SCALE);
        confptr[0] = MY_CONF_DISP;
        
        //if(y==0 && x==8)
        //    printf("[y=%d, x=%d]: d_tmp=%d, Sp[d]=%d, d=%d.\n", y, x, d_tmp, Sbuf[d_tmp], d);
        //if(y==0 && x==1)
        //    printf("[y=%d, x=%d, d=0]: Sp[0]=%d, Sp[1]=%d.\n", y, x, Sbuf[0], Sbuf[1]);
       
        for(d=0; d<D; d++){
            if(Sbuf[d]*(100-confRatio) < minS*100 && std::abs(bestDisp-d) > 1)
                break;
        }
        if(d<D){
            confptr[0] = MY_MED_DISP;
        }

        for(d=0; d<D; d++){
            if(Sbuf[d]*(100-uniquenessRatio) < minS*100 && std::abs(bestDisp-d) > 1)
                break;
        }
        if(d<D){
            confptr[0] = MY_INVALID_DISP;
        }

        if(bestDisp == 0)
            confptr[0] = MY_MED_DISP;

        if(bestDisp == D-1)
            confptr[0] = MY_INVALID_DISP; 
    }

}

template <typename T_src, typename T_dst>
__global__ void pre_consistance_check(T_src* d_disp1_tmp, T_dst* d_disp1cost_tmp, T_dst* d_disp2cost, T_src* d_disp2, int width, int height, int minX1, int minD) {

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(y < height){
        T_src d;
        T_dst minS;
        T_dst* disp2cost = d_disp2cost + y*width;
        T_src* disp2ptr = d_disp2 + y*width;
        for(int x = 0; x < width; ++x){
            d = d_disp1_tmp[y*width + x];
            minS = d_disp1cost_tmp[y*width + x];

            int _x2 = x + minX1 - d - minD;
            if( disp2cost[_x2] > minS ){
                disp2cost[_x2] = minS;
                disp2ptr[_x2] = d + minD;
            }
        }
    }

}

template <typename T_src, typename T_srcdst, typename T_dst>
__global__ void pre_consistance_check_tran(T_src* d_disp1_tmp, T_srcdst* d_disp1cost_tmp, T_srcdst* d_disp2cost, T_dst* d_disp2, int width, int height, int minX1, int minD) {

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(y < height){
        T_src d;
        T_srcdst minS;
        T_srcdst* disp2cost = d_disp2cost + y;
        T_dst* disp2ptr = d_disp2 + y;
        for(int x = 0; x < width; ++x){
            d = d_disp1_tmp[x*height + y];
            minS = d_disp1cost_tmp[x*height + y]; 

            int _x2 = x + minX1 - d - minD;
            if( disp2cost[_x2*height] > minS ){
                disp2cost[_x2*height] = minS;
                disp2ptr[_x2*height] = d + minD;
            }
       /* 
            if( y == 0 && x == 0)
                printf("[y=%d, x=%d]: d = %d, _x2=%d, sum=%d, minS=%d, *disp2cost=%d, *disp2ptr = %d.\n", y, x, d, _x2, d+minD, minS, disp2cost[_x2*height], disp2ptr[_x2*height]);
      */  
        }
    }
}

template <typename T_srcdst, typename T_dst>
__global__ void consistance_check(T_srcdst* d_disp2, T_srcdst* d_disp1, T_dst* d_udispconf, int width, int height, int INVALID_DISP_SCALED, int minD, int sdisp2MaxDiff){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x < width && y < height){
        T_srcdst* disp1ptr = d_disp1 + y * width + x;
        T_dst* confptr = d_udispconf + y * width + x;
        //row ptr for d_disp2
        T_srcdst* disp2ptr = d_disp2 + y * width;

        int d1 = disp1ptr[0];
        if( d1 == INVALID_DISP_SCALED){
            confptr[0] = MY_INVALID_DISP;
            return;
        }

        int _d = d1 >> DISP_SHIFT;
        int d_ = (d1 + DISP_SCALE - 1) >> DISP_SHIFT;
        int _x = x - _d, x_ = x - d_;
        if(0<= _x && _x <width && disp2ptr[_x]>=minD && std::abs(disp2ptr[_x] - _d)>sdisp2MaxDiff &&            0<= x_ && x_ <width && disp2ptr[x_]>=minD && std::abs(disp2ptr[x_] - d_)>sdisp2MaxDiff){
            disp1ptr[0] = (T_srcdst) INVALID_DISP_SCALED;
            confptr[0] = MY_INVALID_DISP;
        }

    }
}

#endif 