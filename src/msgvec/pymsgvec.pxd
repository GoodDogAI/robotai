#!python
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "msgvec.h":
    cdef cppclass MsgVec:
        MsgVec(const string &jsonConfig) except +
        size_t obs_size()
        size_t act_size()

        void input(const vector[uchar] &bytes)

