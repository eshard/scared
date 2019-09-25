import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef np.ndarray[np.float32_t, ndim=1] _c_find_peaks(np.ndarray[np.float32_t, ndim=1] X, int minx, float minh):
    tmp = np.r_[True, X[1:] >= X[:-1]] & np.r_[X[:-1] >= X[1:], True]
    cdef np.ndarray[np.float32_t, ndim=1] maximas = (np.hstack(np.where((X>=minh) & (tmp)))).astype('float32')
    cdef int lm = len(maximas)
    cdef int i, p, mi, mp

    for i in  range(0,lm):
        p=i
        while p<lm-1 and abs(maximas[i]-maximas[p+1])<minx:
            p=p+1
            mi = <int>maximas[i]
            mp = <int>maximas[p]
            if X[mi] < X[mp]:
                maximas[i] =-1
            else:
                maximas[p]=-1
    return maximas[maximas>-1]

cpdef c_find_peaks(trace, minx, minh):
    return _c_find_peaks(trace, minx, minh)
