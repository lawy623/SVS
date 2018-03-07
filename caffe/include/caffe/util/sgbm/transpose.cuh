#ifndef PPL_INTERNAL_MATRIX_TRANSPOSE_CUH_
#define PPL_INTERNAL_MATRIX_TRANSPOSE_CUH_

//#if defined (PPL_USE_CUDA)
inline int divUp(int a, int b){
    return int(ceil(float(a) / float(b)));
}

template <int BS, typename T_src, typename T_dst>
__global__ void transposeKernel(int numRows, int numCols, const T_src* src, T_dst* dst) {
    __shared__ T_src tile[BS][BS+1];

    int blockIdx_x, blockIdx_y;
    if (gridDim.x == gridDim.y) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    int xIndex = blockIdx_x * BS + threadIdx.x;
    int yIndex = blockIdx_y * BS + threadIdx.y;

    if (xIndex < numCols) {
        if (yIndex < numRows) {
            tile[threadIdx.y][threadIdx.x] = src[yIndex*numCols + xIndex];
        }
    }

    __syncthreads();

    xIndex = blockIdx_y * BS + threadIdx.x;
    yIndex = blockIdx_x * BS + threadIdx.y;

    if (xIndex < numRows) {
        if (yIndex < numCols) {
            dst[yIndex*numRows + xIndex] = tile[threadIdx.x][threadIdx.y];
        }
    }
}

template <int BS, typename T_src, typename T_dst>
__host__ void transpose(int device, cudaStream_t stream, int numRows, int numCols, const T_src* src, T_dst *dst) {
    cudaSetDevice(device);
    dim3 block(BS, BS);
    dim3 grid(divUp(numCols, BS), divUp(numRows, BS));

    transposeKernel<BS, T_src, T_dst><<<grid, block, 0, stream>>>(numRows, numCols, src, dst);
}

template <typename T>
inline __host__ void transpose(int device, cudaStream_t stream, int numRows, int numCols, const T* src, T *dst) {
    transpose<16, T, T>(device, stream, numRows, numCols, src, dst);
}

//#endif

#endif
