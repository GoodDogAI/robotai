#!python
#cython: language_level=3

from libcpp.string cimport string

cdef extern from "msgvec.h":
    cdef cppclass MsgVec:
        MsgVec(const string &jsonConfig) except +
        size_t obsSize()
        size_t actSize()

