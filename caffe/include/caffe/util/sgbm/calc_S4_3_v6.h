#ifndef CALC_S4_3_V6_H__
#define CALC_S4_3_V6_H__

#ifndef CUDAMIN
#define cumin(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

template<int num_D>
__global__ void calcSstep1(short* d_Sbuf, const short* __restrict__ d_Cbuf, short* d_Lr0, const short* __restrict__ d_Lr1, short* d_minLr0, const short* __restrict__ d_minLr1, int y, int width1, int D, int batch, int P1, int P2){
    
    int costBufSize = width1 * num_D; // one row of d_Cbuf
    //int D2 = D+2; // 2 denotes two padding disparity values: d=-1 and d=D
    short MAX_COST = SHRT_MAX;
   // extern __shared__ CostType cbuf[];  // the C cost array of one pixel
   // extern __shared__ short minLr0_arr[], minLr1_arr[]; //one direction and length=width1+2
    __shared__ short minLr_p1_arr[num_D], minLr_p2_arr[num_D], minLr_p3_arr[num_D];
    __shared__ short Lr_p1_arr[num_D+2], Lr_p2_arr[num_D+2], Lr_p3_arr[num_D+2]; //one direction and length=D2*(width1+2)
    
    short* Lr0 = d_Lr0; // (h=1, w, r=3, D)--three directions
    short* minLr0 = d_minLr0;
    const short* __restrict__ Lr1 = d_Lr1;
    const short* __restrict__ minLr1 = d_minLr1;

/*    
    if(blockIdx.x == 0 && threadIdx.x ==0){
        printf("0: minLr0 = %p, minLr1 = %p, Lr0 = %p, Lr1 = %p.\n", minLr0, minLr1, Lr0, Lr1);
    }
    if(blockIdx.x == 1 && threadIdx.x ==0){
        printf("1: minLr0 = %p, minLr1 = %p, Lr0 = %p, Lr1 = %p.\n", minLr0, minLr1, Lr0, Lr1);
    }
    if(blockIdx.x == 2 && threadIdx.x ==0){
        printf("2: minLr0 = %p, minLr1 = %p, Lr0 = %p, Lr1 = %p.\n", minLr0, minLr1, Lr0, Lr1);
    }
    if(blockIdx.x == 3 && threadIdx.x ==0){
        printf("3: minLr0 = %p, minLr1 = %p, Lr0 = %p, Lr1 = %p.\n", minLr0, minLr1, Lr0, Lr1);
    }   */
         
    // cbuf[threadIdx.x] = 0;
   // __syncthreads();

#pragma unroll
        for(int i =0, x=blockIdx.x*batch; x < width1 && i < batch; x++, i++){
            // load the C cost values
            //short Cpd =  d_Cbuf[y*costBufSize+x*num_D+threadIdx.x];
            short Cpd =  d_Cbuf[y*costBufSize+x*num_D+threadIdx.x] + P2;
            short* Sp = d_Sbuf + y*costBufSize + x*num_D ; 

            short L1, L2, L3; // minLx = MAX_COST;
            int xd = (x+1)*num_D; //note border

            short delta1 = minLr1[x*3] + P2;  //(x+1)*3 - 3
            short delta2 = minLr1[x*3+4] + P2;  //(x+1)*3 + 1
            short delta3 = minLr1[x*3+8] + P2;  //(x+1)*3 + 3 + 2
            
            //if(threadIdx.x ==0) {
            Lr_p1_arr[0] = MAX_COST;
            Lr_p2_arr[0] = MAX_COST;
            Lr_p3_arr[0] = MAX_COST;
            //}

            Lr_p1_arr[threadIdx.x+1] = Lr1[xd*3 - 3*num_D + threadIdx.x];
            Lr_p2_arr[threadIdx.x+1] = Lr1[xd*3 + num_D + threadIdx.x];
            Lr_p3_arr[threadIdx.x+1] = Lr1[xd*3 + 5*num_D + threadIdx.x]; // xd*3+3*num_D+2*num_D+threadIdx.x;
           /* 
            if ( y == 2 && x == 224 & threadIdx.x == 0){
                printf("ptr0=%p, ptr1=%p, ptr2=%p.\n", &Lr1[xd*3-3*D+threadIdx.x], &Lr1[xd*3+D+threadIdx.x], &Lr1[xd*3+3*D+2*D+threadIdx.x]);
                printf("Vptr0=%d, Vptr1=%d, Vptr2=%d.\n", Lr1[xd*3-3*D+threadIdx.x], Lr1[xd*3+D+threadIdx.x], Lr1[xd*3+3*D+2*D+threadIdx.x]);
                printf("Lr_p0[0]=%d, Lr_p1[0]=%d, Lr_p2=%d.\n", Lr_p1_arr[threadIdx.x+1], Lr_p2_arr[threadIdx.x+1], Lr_p3_arr[threadIdx.x+1]);
                printf("Lr_p0[0]=%p, Lr_p1[0]=%p, Lr_p2=%p.\n", &Lr_p1_arr[threadIdx.x+1], &Lr_p2_arr[threadIdx.x+1], &Lr_p3_arr[threadIdx.x+1]);
            } */ 

           // if(threadIdx.x == 0){
            Lr_p1_arr[num_D+1] = MAX_COST;
            Lr_p2_arr[num_D+1] = MAX_COST;
            Lr_p3_arr[num_D+1] = MAX_COST;
            //}
            __syncthreads();
        
        /*    Lx = Cpd + min( Lr_px[0], min(Lr_px[-1] + P1, 
                        min(Lr_px[1] + P1, delta))) - delta;  */

            L1 = cumin( Lr_p1_arr[threadIdx.x+2] + P1 , delta1);
            L1 = cumin( L1 , Lr_p1_arr[threadIdx.x] + P1);
            L1 = cumin( L1 , Lr_p1_arr[threadIdx.x+1]);
            L1 = L1 + Cpd - delta1;
            Lr0[xd*3+threadIdx.x] = L1;
            
            L2 = cumin( Lr_p2_arr[threadIdx.x+2] + P1 , delta2);
            L2 = cumin( L2 , Lr_p2_arr[threadIdx.x] + P1);
            L2 = cumin( L2 , Lr_p2_arr[threadIdx.x+1]);
            L2 = L2 + Cpd - delta2;
            Lr0[xd*3+num_D+threadIdx.x] = L2;

            L3 = cumin( Lr_p3_arr[threadIdx.x+2] + P1 , delta3);
            L3 = cumin( L3 , Lr_p3_arr[threadIdx.x] + P1);
            L3 = cumin( L3 , Lr_p3_arr[threadIdx.x+1]);
            L3 = L3 + Cpd - delta3;
            Lr0[xd*3+2*num_D+threadIdx.x] = L3;
            
           
            /*
            if ( y == 4 && x == 94){
                //printf("ptr0=%p, ptr1=%p, ptr2=%p.\n", &Lr1[xd*3-3*D+threadIdx.x], &Lr1[xd*3+D+threadIdx.x], &Lr1[xd*3+3*D+2*D+threadIdx.x]);
                //printf("Vptr0=%d, Vptr1=%d, Vptr2=%d.\n", Lr1[xd*3-3*D+threadIdx.x], Lr1[xd*3+D+threadIdx.x], Lr1[xd*3+3*D+2*D+threadIdx.x]);
              //  printf("[%d, %d, 0]: Lr_px[1]=%d, Lr_px[-1]=%d, Lr_px[0]=%d, Cpd=%d, delta=%d, sum=%d.\n", y, x, Lr_p1_arr[threadIdx.x+2], Lr_p1_arr[threadIdx.x], Lr_p1_arr[threadIdx.x+1], Cpd, delta1, L1);
                printf("[%d, %d, 2]: Lr_px[1]=%d, Lr_px[-1]=%d, Lr_px[0]=%d, Cpd=%d, delta=%d, sum=%d.\n", y, x, Lr_p3_arr[threadIdx.x+2], Lr_p3_arr[threadIdx.x], Lr_p3_arr[threadIdx.x+1], Cpd, delta3, L3);
              //  printf("[%d, %d, 1]: Lr_px[1]=%d, Lr_px[-1]=%d, Lr_px[0]=%d, Cpd=%d, delta=%d, sum=%d.\n", y, x, Lr_p2_arr[threadIdx.x+2], Lr_p2_arr[threadIdx.x], Lr_p2_arr[threadIdx.x+1], Cpd, delta2, L2);
            } 
            */

            minLr_p1_arr[threadIdx.x] = L1;
            __syncthreads();
            
            for(int i = num_D; i >= 2; i = i / 2) {
            if (threadIdx.x < i/2) minLr_p1_arr[threadIdx.x] = cumin(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x + i/2 ]);
            __syncthreads(); 
            }
            /*
            if (threadIdx.x < 16) minLr_p1_arr[threadIdx.x] = min(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x+16]);
            __syncthreads(); 
            if (threadIdx.x < 8) minLr_p1_arr[threadIdx.x] = min(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x+8]);
            __syncthreads(); 
            if (threadIdx.x < 4) minLr_p1_arr[threadIdx.x] = min(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x+4]);
            __syncthreads(); 
            if (threadIdx.x < 2) minLr_p1_arr[threadIdx.x] = min(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x+2]);
            __syncthreads(); 
            if (threadIdx.x < 1) minLr_p1_arr[threadIdx.x] = min(minLr_p1_arr[threadIdx.x], minLr_p1_arr[threadIdx.x+1]);
            __syncthreads(); 
            */
            if (threadIdx.x == 0)
                minLr0[(x+1)*3] = minLr_p1_arr[0];
           // __syncthreads();

            //L2
            minLr_p2_arr[threadIdx.x] = L2;
            __syncthreads();
            for(int i = num_D; i >= 2; i = i / 2) {
                if (threadIdx.x < i/2) minLr_p2_arr[threadIdx.x] = cumin(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x + i/2]);
                __syncthreads();
            }
            /*
            if (threadIdx.x < 16) minLr_p2_arr[threadIdx.x] = min(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x+16]);
            __syncthreads(); 
            if (threadIdx.x < 8) minLr_p2_arr[threadIdx.x] = min(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x+8]);
            __syncthreads(); 
            if (threadIdx.x < 4) minLr_p2_arr[threadIdx.x] = min(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x+4]);
            __syncthreads(); 
            if (threadIdx.x < 2) minLr_p2_arr[threadIdx.x] = min(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x+2]);
            __syncthreads(); 
            if (threadIdx.x < 1) minLr_p2_arr[threadIdx.x] = min(minLr_p2_arr[threadIdx.x], minLr_p2_arr[threadIdx.x+1]);
            __syncthreads(); 
            */

            if (threadIdx.x == 0)
                minLr0[(x+1)*3+1] = minLr_p2_arr[0];
            //__syncthreads();

            //L3
            minLr_p3_arr[threadIdx.x] = L3;
            __syncthreads();
            for(int i = num_D; i >= 2; i = i/2) {
                if (threadIdx.x < i/2) minLr_p3_arr[threadIdx.x] = cumin(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x + i/2]);
                __syncthreads(); 
            }

            /*
            if (threadIdx.x < 16) minLr_p3_arr[threadIdx.x] = min(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x+16]);
            __syncthreads(); 
            if (threadIdx.x < 8) minLr_p3_arr[threadIdx.x] = min(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x+8]);
            __syncthreads(); 
            if (threadIdx.x < 4) minLr_p3_arr[threadIdx.x] = min(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x+4]);
            __syncthreads(); 
            if (threadIdx.x < 2) minLr_p3_arr[threadIdx.x] = min(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x+2]);
            __syncthreads(); 
            if (threadIdx.x < 1) minLr_p3_arr[threadIdx.x] = min(minLr_p3_arr[threadIdx.x], minLr_p3_arr[threadIdx.x+1]);
            __syncthreads(); 
            */

            if (threadIdx.x == 0)
                minLr0[(x+1)*3+2] = minLr_p3_arr[0];
           // __syncthreads();
            
            Sp[threadIdx.x] = Sp[threadIdx.x] + L1 + L2 + L3;
            __syncthreads();
        }
        
}

#endif