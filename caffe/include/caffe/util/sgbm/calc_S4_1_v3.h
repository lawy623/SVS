#ifndef CALC_S4_1_V3_H__
#define CALC_S4_1_V3_H__

#ifndef CUDAMIN
#define cumin(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

template<int T, int batch>
__global__ void calcSstep2_left2right(short* d_Sbuf, const short* __restrict__ d_Cbuf, int width1, int height, int D, int P1, int P2, short MAX_COST){
    
    int costBufSize = width1 * D; // one row of d_Cbuf

   // int batch = 2;
    __shared__ short Lr_p_arr[batch][T]; // Lr_px_arr[66];  //one direction and length=D2*(width1+2)
    short Lr1 = 0;
    short minLr1 = 0; 
         
    short Cpd, Lx, deltax;
    short *Sp;
    //Lr1[threadIdx.x] = 0;
    //__syncthreads();
    short Lr_px_arr0 = (threadIdx.x == 0) ? MAX_COST : 0;
    short Lr_px_arr1 = 0;
    short Lr_px_arr2 = (threadIdx.x == (D-1)) ? MAX_COST : 0;
    //__syncthreads();
    for(int x=0; x < width1; x++){
        // load the C cost values
        int y = blockIdx.x*batch + threadIdx.y;
        
        if(y<height){
            Cpd =  d_Cbuf[y*costBufSize+x*D+threadIdx.x];
            Sp = d_Sbuf + y*costBufSize + x*D ; 

            deltax = minLr1 + P2; //minLr1[x*3 + threadIdx.y*4] + P2;

            Lx = cumin(Lr_px_arr0, Lr_px_arr2);
            Lx = cumin(Lx+P1, Lr_px_arr1);
            Lr1 = (Lx - deltax < 0) ? Lx + Cpd - deltax : Cpd;

            Lr_p_arr[threadIdx.y][threadIdx.x] = Lr1; 
            Sp[threadIdx.x] = Sp[threadIdx.x] + Lr1;
            __syncthreads();
        
            Lr_px_arr0 = (threadIdx.x == 0) ? MAX_COST : Lr_p_arr[threadIdx.y][threadIdx.x-1];
            Lr_px_arr1 = Lr1; //Lr_p_arr[threadIdx.y][threadIdx.x];
            Lr_px_arr2 = (threadIdx.x == (D-1) ) ? MAX_COST : Lr_p_arr[threadIdx.y][threadIdx.x+1];
            __syncthreads();
        
            for(int i = T / 2; i >= 1; i = i / 2) {
                if (threadIdx.x < i) Lr_p_arr[threadIdx.y][threadIdx.x] = cumin(Lr_p_arr[threadIdx.y][threadIdx.x], Lr_p_arr[threadIdx.y][threadIdx.x + i]);
                __syncthreads(); 
            }
       
            minLr1 = Lr_p_arr[threadIdx.y][0];
        }    
        __syncthreads();
    }
}

template<int T, int batch>
__global__ void calcSstep2_right2left(short* d_Sbuf, const short* __restrict__ d_Cbuf, int width1, int height, int D, int P1, int P2, short MAX_COST){
    
    int costBufSize = width1 * D; // one row of d_Cbuf

   // int batch = 2;
    __shared__ short Lr_p_arr[batch][T]; // Lr_px_arr[66];  //one direction and length=D2*(width1+2)
    short Lr1 = 0;
    short minLr1 = 0; 
         
    short Cpd, Lx, deltax;
    short *Sp;
    //Lr1[threadIdx.x] = 0;
    //__syncthreads();
    short Lr_px_arr0 = (threadIdx.x == 0) ? MAX_COST : 0;
    short Lr_px_arr1 = 0;
    short Lr_px_arr2 = (threadIdx.x == (D-1)) ? MAX_COST : 0;
    //__syncthreads();
    for(int x=width1-1; x > -1; x--){
        // load the C cost values
        int y = blockIdx.x*batch + threadIdx.y;
        
        if(y<height){
            Cpd =  d_Cbuf[y*costBufSize+x*D+threadIdx.x];
            Sp = d_Sbuf + y*costBufSize + x*D ; 

            deltax = minLr1 + P2; //minLr1[x*3 + threadIdx.y*4] + P2;

            Lx = min(Lr_px_arr0, Lr_px_arr2);
            Lx = min(Lx+P1, Lr_px_arr1);
            Lr1 = (Lx - deltax < 0) ? Lx + Cpd - deltax : Cpd;

            Lr_p_arr[threadIdx.y][threadIdx.x] = Lr1; 
            Sp[threadIdx.x] = Sp[threadIdx.x] + Lr1;
            __syncthreads();
        
            Lr_px_arr0 = (threadIdx.x == 0) ? MAX_COST : Lr_p_arr[threadIdx.y][threadIdx.x-1];
            Lr_px_arr1 = Lr1; //Lr_p_arr[threadIdx.y][threadIdx.x];
            Lr_px_arr2 = (threadIdx.x == (D-1) ) ? MAX_COST : Lr_p_arr[threadIdx.y][threadIdx.x+1];
            __syncthreads();
        
            for(int i = T / 2; i >= 1; i = i / 2) {
                if (threadIdx.x < i) Lr_p_arr[threadIdx.y][threadIdx.x] = min(Lr_p_arr[threadIdx.y][threadIdx.x], Lr_p_arr[threadIdx.y][threadIdx.x + i]);
                __syncthreads(); 
            }
            
            minLr1 = Lr_p_arr[threadIdx.y][0];
        }
        __syncthreads();
    }
}

#endif