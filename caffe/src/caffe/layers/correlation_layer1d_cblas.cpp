#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/correlation_1d_cblas_layer.hpp"

using namespace std;
namespace caffe
{

template <typename Dtype>
Dtype max(Dtype a, Dtype b)
{
  return a > b ? a : b;
}

template <typename Dtype>
Dtype min(Dtype a, Dtype b)
{
  return a > b ? b : a;
}

template <typename Dtype>
void Correlation1DCblasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
{
  // Configure the kernel size, padding, stride, and inputs.
  CorrelationParameter corr_param = this->layer_param_.correlation_param();

  CHECK(corr_param.has_kernel_size()) << "Filter kernel_size is not set";
  CHECK(corr_param.has_max_displacement()) << "Max displacement is required.";

  kernel_size_ = corr_param.kernel_size();
  if (kernel_size_ % 2 == 0)
    LOG(FATAL) << "Odd kernel size required";

  max_displacement_ = corr_param.max_displacement();
  pad_size_ = corr_param.pad();
  stride1_ = corr_param.stride_1();
  stride2_ = corr_param.stride_2();
  pad_shift_ = corr_param.pad_shift();
  /*single_direction_ = corr_param.single_direction();
  if(single_direction_ < -1 || single_direction_ > 1) LOG(FATAL) << "single_direction must be -1 (left), 0 (off), or 1 (right)";*/

  do_abs_ = corr_param.do_abs();

  corr_type_ = corr_param.correlation_type();

  LOG(INFO) << "Kernel Size: " << kernel_size_;
  LOG(INFO) << "Stride 1: " << stride1_;
  LOG(INFO) << "Stride 2: " << stride2_;
  LOG(INFO) << "Max Displacement: " << max_displacement_;
}

template <typename Dtype>
void Correlation1DCblasLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top)
{

  num_ = bottom[0]->num();

  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Both bottom blobs must have same width";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Both bottom blobs must have same height";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Both bottom blobs must have same number of channels";

  int bottomchannels = bottom[0]->channels();

  // Size computation
  kernel_radius_ = (kernel_size_ - 1) / 2;           //size of unreachable border region (on each side)
  border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)

  int paddedbottomheight = bottom[0]->height() + 2 * kernel_radius_; // Jiahao modified
  int paddedbottomwidth = bottom[0]->width() + 2 * pad_size_;

  top_width_ = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1_);
  top_height_ = ceil((float)(paddedbottomheight - kernel_radius_ * 2) / (float)stride1_);

  CHECK_GE(top_width_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  CHECK_GE(top_height_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";

  // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
  neighborhood_grid_radius_ = max_displacement_ / stride2_;
  neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

  /*if(single_direction_ != 0) {
    neighborhood_grid_width_ = neighborhood_grid_radius_ + 1;
  } else {
    neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
  }*/

  // Top Channels amount to displacement combinations in X direction only!!
  top_channels_ = neighborhood_grid_width_; // Same, because 1D X-correlation

  //Reshape top
  top[0]->Reshape(num_, top_channels_, top_height_, top_width_);

  // rbots (These are the blobs that store the padded and dimension rearranged data
  rbot1_.reset(new Blob<Dtype>());
  rbot2_.reset(new Blob<Dtype>());
  rbot1_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  rbot2_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);

  rtopdiff_.reset(new Blob<Dtype>());
  rtopdiff_->Reshape(num_, top_height_, top_width_, top_channels_);
}

template <typename Dtype>
void blob_rearrange_kernel2(const Dtype *in, Dtype *out, int num, int channels, int width, int height, int widthheight, int padding, int padshift, int pwidthheight, int padding_kernel)
{
  for (int n = 0; n < num; n++)
  {

    for (int ch = 0; ch < channels; ch++)
    {

      for (int xy = 0; xy < widthheight; xy++)
      {

        Dtype value = in[(n * channels + ch) * widthheight + xy];
        int xpad = (xy % width + padding - padshift);
        int ypad = (xy / width + padding_kernel);
        int xypad = ypad * (width + 2 * padding) + xpad;

        out[(n * pwidthheight + xypad) * channels + ch] = value;
      }
    }
  }
}

// == Correlation Kernel
template <typename Dtype>
void CorrelateData(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
                   int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
                   int bottomwidth, int bottomheight, int bottomchannels,
                   const Dtype *bottom0, const Dtype *bottom1, Dtype *top)
{
  // Compute correlation
  const int sumelems = bottomchannels;
  float a = 1 / (float)sumelems;
  int mheight = topwidth;
  int mwidth = topwidth + neighborhood_grid_width - 1;
  Dtype *m = new Dtype[mwidth * mheight];
  // ofstream out("/home/sensetime/a.txt");
  // ofstream out2("/home/sensetime/a2.txt");
  // ofstream out3("/home/sensetime/a3.txt");
  // ofstream out4("/home/sensetime/bottom1.txt");
  // out << "mheight: " << mheight << endl;
  // out << "mwidth: " << mwidth << endl;
  // out << "bottomchannels: " << bottomchannels << endl;
  // out << "num: " << num << endl;
  // out << "topheight: " << topheight << endl;
  // out << "topwidth: " << topwidth << endl;

  for (int item = 0; item < num; item++)
  {
    int itemtopcount = item * topcount;
    for (int y = 0; y < topheight; y++)
    {
      int idx0 = (item * bottomheight + y) * topwidth * bottomchannels;
      int idx1 = (item * bottomheight + y) * bottomwidth * bottomchannels;
      // for (int i = 0; i < mwidth * bottomchannels; i++)
      // {
      //   out4 << setfill('0') << setw(5) << setprecision(2) << std::fixed << bottom1[i];
      //   if (i % bottomchannels == (bottomchannels - 1))
      //     out4 << endl;
      //   else
      //     out4 << ",";
      // }
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            mheight, mwidth, bottomchannels,
                            a, bottom0 + idx0, bottom1 + idx1, 0, m);
      // for (int i = 0; i < mheight * mwidth; i++)
      // {
      //   out3 << setfill('0') << setw(5) << setprecision(2) << std::fixed << m[i];
      //   if (i % mwidth == (mwidth - 1))
      //     out3 << endl;
      //   else
      //     out3 << ",";
      // }
      //system("pause");

      for (int x = 0; x < topwidth; x++)
      {
        for (int top_channel = 0; top_channel < topchannels; top_channel++)
        {
          int index = ((top_channel * topheight + y) * topwidth) + x;

          top[index + itemtopcount] = m[x * mwidth + x + top_channel];
          // out << index << "   " << x * mwidth + x + top_channel << endl;
        }
      }
    }
    // for (int i = 0; i < topheight * topchannels * topwidth; i++)
    // {
    //   out2 << setfill('0') << setw(5) << setprecision(2) << std::fixed << top[itemtopcount + i];
    //   if (i % topwidth == (topwidth - 1))
    //     out2 << endl;
    //   else
    //     out2 << ",";
    //   if (i % topheight * topwidth == (topheight * topwidth - 1))
    //     out2 << endl;
    // }
  }
}

// // == Correlation Backward Pass Kernel (For Blob 0)
// template <typename Dtype>
// void CorrelateDataBackward0(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
//   int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
//   int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
//   Dtype *bottom0diff, const Dtype *bottom1, const Dtype *topdiff)
// {
//   for (int index= 0; index < nthreads; index++){
//     int n = index % bottomchannels; //channels
//     int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
//     int m = (index / bottomchannels / bottomwidth) % bottomheight + kernel_radius; //h-pos, Jiahao modified

//     //Get X,Y ranges and clamp
//     // round_off is a trick to enable integer division with ceil, even for negative numbers
//     // We use a large offset, for the inner part not to become negative.
//     const int round_off = ROUND_OFF;
//     const int round_off_s1 = stride1 * round_off;

//     // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
//     int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
//     int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1

//     // Same here:
//     int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
//     int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1

//     Dtype sum = 0;
//     if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
//     {
//         xmin = max(0,xmin);
//         xmax = min(topwidth-1,xmax);

//         ymin = max(0,ymin);
//         ymax = min(topheight-1,ymax);

//         {
//           for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

//             // Get bottom1 data:
//             int s2o = stride2 * o;
//             int idxbot1 = ((item * pbottomheight + m) * pbottomwidth + (l+s2o)) * bottomchannels + n;
//             Dtype bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m,n]

//             // Index offset for topdiff in following loops:
//             int op = (o-x_shift); // index [o,p]
//             int idxopoffset = (item * topchannels + op);

//             for(int y = ymin; y <= ymax; y++) {
//               for(int x = xmin; x <= xmax; x++) {
//                 int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
//                 sum += topdiff[idxtopdiff] * bot1tmp;
//               }
//             }
//           }
//         }
//     }
//     const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
//     const int bot0index = ((n * bottomheight) + (m-kernel_radius)) * bottomwidth + (l-pad_size); // Jiahao modified
//     bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
//   }

// }

// // == Correlation Backward Pass Kernel (For Blob 1)
// template <typename Dtype>
// void CorrelateDataBackward1(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
//   int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
//   int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
//   const Dtype *bottom0, Dtype *bottom1diff, const Dtype *topdiff)
// {
//   for(int index =0; index< nthreads; index++) {
//     //int l = index % bottomwidth + pad_size; //w-pos
//     //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
//     //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
//     int n = index % bottomchannels; //channels
//     int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
//     int m = (index / bottomchannels / bottomwidth) % bottomheight + kernel_radius; //h-pos, Jiahao modified

//     // round_off is a trick to enable integer division with ceil, even for negative numbers
//     // We use a large offset, for the inner part not to become negative.
//     const int round_off = ROUND_OFF;
//     const int round_off_s1 = stride1 * round_off;

//     Dtype sum = 0;
//     {

//       for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

//         int s2o = stride2 * o;

//         //Get X,Y ranges and clamp
//         // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
//         int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
//         int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

//         // Same here:
//         int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
//         int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - 0) / stride1

//         if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
//         {
//             xmin = max(0,xmin);
//             xmax = min(topwidth-1,xmax);

//             ymin = max(0,ymin);
//             ymax = min(topheight-1,ymax);

//             // Get bottom0 data:
//             int idxbot0 = ((item * pbottomheight + m) * pbottomwidth + (l-s2o)) * bottomchannels + n;
//             Dtype bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m,n]

//             // Index offset for topdiff in following loops:
//             int op = (o-x_shift); // index [o,p]
//             int idxOpOffset = (item * topchannels + op);

//             for(int y = ymin; y <= ymax; y++) {
//               for(int x = xmin; x <= xmax; x++) {
//                 int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
//                 sum += topdiff[idxtopdiff] * bot0tmp;
//               }
//             }
//         }
//       }
//     }
//     const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
//     const int bot1index = ((n * bottomheight) + (m-kernel_radius)) * bottomwidth + (l-pad_size); // Jiahao modified
//     bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
//   }

// }

// == Correlation Kernel Subtraction
template <typename Dtype>
void CorrelateDataSubtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels, int topcount,
                           int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
                           int bottomwidth, int bottomheight, int bottomchannels,
                           const Dtype *bottom0, const Dtype *bottom1, Dtype *top)
{
  for (int index = 0; index < nthreads; index++)
  {
    int x = index % topwidth;                             //w-pos
    int y = (index / topwidth) % topheight;               //h-pos
    int c = (index / topwidth / topheight) % topchannels; //channels

    // Offset of patch in image 2
    int s2o = (c % neighborhood_grid_width + x_shift) * stride2;

    // First (upper left) position of kernel center in current neighborhood in image 1
    int x1 = x * stride1 + kernel_radius + max_displacement;
    int y1 = y * stride1 + kernel_radius + 0;

    // Iterate through 3D patch
    Dtype sum = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++)
    { // HEIGHT
      for (int i = -kernel_radius; i <= kernel_radius; i++)
      { // WIDTH
        for (int l = 0; l < bottomchannels; l++)
        { // CHANNELS
          // Calculate position in image 2
          int x2 = x1 + s2o;
          int y2 = y1;

          // Indices in bottom data: (CH=l,W=x2,H=y2,N)
          int idx1 = ((item * bottomheight + y1 + j) * bottomwidth + x1 + i) * bottomchannels + l;
          int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + l;

          // Do the correlation:
          sum += fabsf(bottom0[idx1] - bottom1[idx2]);
        }
      }
    }
    const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bottomchannels;
    top[index + item * topcount] = sum / (float)sumelems;
  }
}

// // == Correlation Backward Pass Kernel (For Blob 0)
// template <typename Dtype>
// void CorrelateDataBackward0Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
//   int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
//   int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
//   Dtype *bottom0diff, const Dtype *bottom0, const Dtype *bottom1, const Dtype *topdiff)
// {
//   for (int index = 0; index < nthreads; index++) {
//     int l = index % bottomwidth + pad_size; //w-pos
//     int m = (index / bottomwidth) % bottomheight + kernel_radius; //h-pos, Jiahao modified
//     int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

//     //Get X,Y ranges and clamp
//     // round_off is a trick to enable integer division with ceil, even for negative numbers
//     // We use a large offset, for the inner part not to become negative.
//     const int round_off = ROUND_OFF;
//     const int round_off_s1 = stride1 * round_off;

//     // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
//     int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
//     int ymin = (m - 2*kernel_radius - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1

//     // Same here:
//     int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
//     int ymax = (m - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1

//     Dtype sum = 0;
//     if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
//     {
//         xmin = max(0,xmin);
//         xmax = min(topwidth-1,xmax);

//         ymin = max(0,ymin);
//         ymax = min(topheight-1,ymax);

//         {
//           for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

//             // Get bottom1 data:
//             int s2o = stride2 * o;
//             int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
//             Dtype bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
//             Dtype bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
//             Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(1.0) : Dtype(-1.0);

//             // Index offset for topdiff in following loops:
//             int op = (o-x_shift); // index [o,p]
//             int idxopoffset = (item * topchannels + op);

//             for(int y = ymin; y <= ymax; y++) {
//               for(int x = xmin; x <= xmax; x++) {
//                 int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
//                 sum += topdiff[idxtopdiff] * sign;
//               }
//             }
//           }
//         }
//     }
//     const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
//     bottom0diff[index + item*bottomcount] = sum / (float)sumelems;
//   }

// }

// // == Correlation Backward Pass Kernel (For Blob 1)
// template <typename Dtype>
// __global__ void CorrelateDataBackward1Subtract(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
//   int max_displacement, int x_shift, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
//   int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
//   const Dtype *bottom0, const Dtype *bottom1, Dtype *bottom1diff, const Dtype *topdiff)
// {
//   for (int index = 0; index < nthreads; index++) {
//     int l = index % bottomwidth + pad_size; //w-pos
//     int m = (index / bottomwidth) % bottomheight + kernel_radius; //h-pos, Jiahao modified
//     int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels

//     // round_off is a trick to enable integer division with ceil, even for negative numbers
//     // We use a large offset, for the inner part not to become negative.
//     const int round_off = ROUND_OFF;
//     const int round_off_s1 = stride1 * round_off;

//     Dtype sum = 0;
//     {
//       for(int o = x_shift; o < x_shift + neighborhood_grid_width; o++) {

//         int s2o = stride2 * o;

//         //Get X,Y ranges and clamp
//         // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
//         int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
//         int ymin = (m - 2*kernel_radius - 0 - 0 + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1

//         // Same here:
//         int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
//         int ymax = (m - 0 - 0 + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

//         if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
//         {
//             xmin = max(0,xmin);
//             xmax = min(topwidth-1,xmax);

//             ymin = max(0,ymin);
//             ymax = min(topheight-1,ymax);

//             // Get bottom0 data:
//             int idxbot = ((item * pbottomheight + (m)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
//             Dtype bot0tmp = bottom0[idxbot]; // bottom0[l+s2o,m,n]
//             Dtype bot1tmp = bottom1[idxbot]; // bottom1[l+s2o,m,n]
//             Dtype sign = (bot0tmp >= bot1tmp) ? Dtype(-1.0) : Dtype(1.0);

//             // Index offset for topdiff in following loops:
//             int op = (o-x_shift); // index [o,p]
//             int idxOpOffset = (item * topchannels + op);

//             for(int y = ymin; y <= ymax; y++) {
//               for(int x = xmin; x <= xmax; x++) {
//                 int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
//                 sum += topdiff[idxtopdiff] * sign;
//               }
//             }
//         }
//       }
//     }
//     const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
//     bottom1diff[index + item*bottomcount] = sum / (float)sumelems;
//   }

// }

template <typename Dtype>
void Correlation1DCblasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top)
{
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);

  const int bnum = bottom[0]->num();
  const int bchannels = bottom[0]->channels();
  const int bheight = bottom[0]->height();
  const int bwidth = bottom[0]->width();
  const int bwidthheight = bwidth * bheight;

  const int topcount = top_width_ * top_height_ * top_channels_;
  const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight + 2 * kernel_radius_); // Jiahao modified
  // time_t t = clock();
  blob_rearrange_kernel2<Dtype>(bottom[0]->cpu_data(), rbot1_->mutable_cpu_data(), bnum, bchannels, bwidth, bheight, bwidthheight, 0, 0, bwidthheight, kernel_radius_);
  blob_rearrange_kernel2<Dtype>(bottom[1]->cpu_data(), rbot2_->mutable_cpu_data(), bnum, bchannels, bwidth, bheight, bwidthheight, pad_size_, pad_shift_, pwidthheight, kernel_radius_);
  // t = clock() - t;
  // cout << "blob_rearrange_kernel2: " << (double)t / CLOCKS_PER_SEC << endl;
  
  const int num = bnum;
  const int channels = bchannels;
  const int height = bheight + 2 * kernel_radius_; // Jiahao modified
  const int width = bwidth + 2 * pad_size_;

  int x_shift = -neighborhood_grid_radius_;
  /*if(single_direction_ == -1) { // to the left
      x_shift = -neighborhood_grid_width_;
    } else if(single_direction_ == 1) { // to the right
      x_shift = 0;
    }
    */
  if (corr_type_ == CorrelationParameter_CorrelationType_MULTIPLY)
  {
    // Correlation1DCblasLayer
    int topThreadCount = topcount;
    // t = clock();
    CorrelateData<Dtype>(topThreadCount, num, top_width_, top_height_, top_channels_, topcount,
                         max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_, kernel_size_,
                         stride1_, stride2_,
                         width, height, channels,
                         rbot1_->cpu_data(), rbot2_->cpu_data(), top[0]->mutable_cpu_data());
    // t = clock() - t;
    // cout << "CorrelateData: " << (double)t / CLOCKS_PER_SEC << endl;
  }
  else if (corr_type_ == CorrelationParameter_CorrelationType_SUBTRACT)
  {
    // Correlation1DCblasLayer
    for (int n = 0; n < num; n++)
    {
      int topThreadCount = topcount;
      CorrelateDataSubtract<Dtype>(topThreadCount, num, n, top_width_, top_height_, top_channels_, topcount,
                                   max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
                                   stride1_, stride2_,
                                   width, height, channels,
                                   rbot1_->cpu_data(), rbot2_->cpu_data(), top[0]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void Correlation1DCblasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                             const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom)
{
  NOT_IMPLEMENTED;
  // // Get top diff, compute bottom diff
  // const Dtype* top_diff = top[0]->cpu_diff();

  // Dtype* bottom0_diff = bottom[0]->mutable_cpu_diff();
  // Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();

  // const Dtype* bottom0_data = bottom[0]->cpu_data();
  // const Dtype* bottom1_data = bottom[1]->cpu_data();

  // const int num = bottom[0]->num();
  // const int channels = bottom[0]->channels();
  // const int height = bottom[0]->height();
  // const int width = bottom[0]->width();

  // const int paddedheight = height + 2*kernel_radius_; // Jiahao modified
  // const int paddedwidth = width + 2*pad_size_;

  // const int bottomcount = channels * height * width;

  // int botThreadCount = bottomcount;

  // // CorrelationLayerBackward

  // bottom0_diff = bottom[0]->mutable_cpu_diff();
  // bottom1_diff = bottom[1]->mutable_cpu_diff();

  // int x_shift = - neighborhood_grid_radius_;
  // /*if(single_direction_ == -1) { // to the left
  //   x_shift = -neighborhood_grid_width_;
  // } else if(single_direction_ == 1) { // to the right
  //   x_shift = 0;
  // }
  // */

  // if(corr_type_ == CorrelationParameter_CorrelationType_MULTIPLY)
  // {

  //     // == Run kernel Backward 0
  //    // dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
  //    // dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK);
  //     const int buffer_size_backw0 = ((int)ceil((float)(2 * kernel_radius_) / (float)stride1_) + 1) * top_channels_;

  //     // == Run kernel Backward 0
  //     for(int n = 0; n < num; n++) {
  //     //Bottom0:
  //         CorrelateDataBackward0<Dtype>(botThreadCount, num, n, top_width_, top_height_, top_channels_,
  //         max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
  //         stride1_, stride2_,
  //         width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
  //         bottom0_diff, rbot2_->cpu_data(), top_diff
  //         );

  //     //CUDA_POST_KERNEL_CHECK;
  //     }

  //     // == Run kernel Backward 1
  //     for(int n = 0; n < num; n++) {
  //         CorrelateDataBackward1<Dtype>( botThreadCount, num, n, top_width_, top_height_, top_channels_,
  //         max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
  //         stride1_, stride2_,
  //         width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
  //         rbot1_->cpu_data(), bottom1_diff, top_diff
  //         );

  //     CUDA_POST_KERNEL_CHECK;
  //     }

  // }
  // else if(corr_type_ == CorrelationParameter_CorrelationType_SUBTRACT)
  // {
  //     for(int n = 0; n < num; n++) {
  //     //Bottom0:
  //         CorrelateDataBackward0Subtract<Dtype>(botThreadCount, num, n, top_width_, top_height_, top_channels_,
  //         max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
  //         stride1_, stride2_,
  //         width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
  //         bottom0_diff, rbot1_->cpu_data(), rbot2_->cpu_data(), top_diff
  //         );

  //     //CUDA_POST_KERNEL_CHECK;
  //     }

  //     for(int n = 0; n < num; n++) {
  //     //Bottom0:
  //         CorrelateDataBackward1Subtract<Dtype>(botThreadCount, num, n, top_width_, top_height_, top_channels_,
  //         max_displacement_, x_shift, neighborhood_grid_width_, kernel_radius_,
  //         stride1_, stride2_,
  //         width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
  //         rbot1_->cpu_data(), rbot2_->cpu_data(), bottom1_diff, top_diff
  //         );

  //     //CUDA_POST_KERNEL_CHECK;
  //     }
  // }
}

#ifdef CPU_ONLY
STUB_GPU(Correlation1DLayer);
#endif

INSTANTIATE_CLASS(Correlation1DCblasLayer);
REGISTER_LAYER_CLASS(Correlation1DCblas);

} // namespace caffe
