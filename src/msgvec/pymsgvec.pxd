#!python
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "msgvec.h":
    cdef cppclass MsgVec:
        MsgVec(const string &jsonConfig) except +
        size_t obs_size()
        size_t act_size()

        bool input(const vector[uchar] &bytes) except +

