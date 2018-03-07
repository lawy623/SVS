// Copyright 2015 Zhu.Jin Liang

#ifndef CAFFE_UTIL_PRE_DEFINE_H_
#define CAFFE_UTIL_PRE_DEFINE_H_

#include <algorithm>
#include <utility>
#include <vector>

namespace caffe {

#ifndef ELLISION
#		define ELLISION 1e-5
#endif //ifndef ELLISION

#ifndef ABS
#		define	ABS(a)	( ((a) < 0) ? (-(a)) : (a) )
#endif // ifndef ABS

#ifndef isValidCoord
#		define	IsValidCoord(a)  (! ( (ABS(a - (-1))) < (ELLISION) ) )
#endif // ifndef isValidCoord

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

template <typename Dtype>
inline bool IsEqual(Dtype val1, Dtype val2) {
	return (ABS(val1 - val2)) < Dtype(ELLISION);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_PRE_DEFINE_H_