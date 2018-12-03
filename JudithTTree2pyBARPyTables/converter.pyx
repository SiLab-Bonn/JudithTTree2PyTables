# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=2

import numpy as np
cimport numpy as cnp

from numpy cimport ndarray
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from libcpp.vector cimport vector as std_vector

cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error


# declaration imported from C++ source file (C++ header file should be used), do not add source file to setup.py
cdef extern from "converter_src.cpp":
    struct data_row:
        cnp.int64_t event_number
        cnp.uint8_t frame
        cnp.uint16_t column
        cnp.uint16_t row
        cnp.uint16_t charge
        cnp.uint16_t tdc_value
        cnp.uint16_t tdc_time_stamp
        cnp.uint8_t trigger_status
        cnp.uint16_t event_status
        cnp.int64_t pybar_event_number

    void read_tree(const char * input_file, const char * plane, std_vector[data_row] & data) except +


cdef data_type = cnp.dtype([
    ('event_number', np.int64),
    ('frame', np.uint8),
    ('column', np.uint16),
    ('row', np.uint16),
    ('charge', np.uint16),
    ('TDC', np.uint16),
    ('TDC_time_stamp', np.uint16),
    ('trigger_status', np.uint8),
    ('event_status', np.uint16),
    ('pybar_event_number', np.int64)])


# force numpy to take ownership of memory
# http://stackoverflow.com/questions/23872946/force-numpy-ndarray-to-take-ownership-of-its-memory-in-cython
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)


def read_from_root_tree(input_file, plane):
    # vector is initialized here
    cdef std_vector[data_row] my_data

    # call c++ function
    read_tree(< const char * > input_file, < const char * > plane, < std_vector[data_row] & > my_data)

    # Return nothing if selected plane does not exist
    if my_data.size() == 0:
        return

    # if plane exists and data is converted, format data to numpy array
    data_array = hit_data_to_numpy_array(& my_data[0], my_data.size() * sizeof(data_row))
    return data_array


cdef hit_data_to_numpy_array(void * ptr, cnp.npy_intp N):
    # use copy because vector my_data will be deleted when returning read_from_root_tree function
    cdef cnp.ndarray[data_row, ndim = 1] formatted = cnp.PyArray_SimpleNewFromData(1, < cnp.npy_intp * > & N, cnp.NPY_INT8, < void * > ptr).view(data_type).copy()
    formatted.setflags(write=True)  # protect the hit data
    PyArray_ENABLEFLAGS(formatted, cnp.NPY_OWNDATA)
    return formatted
