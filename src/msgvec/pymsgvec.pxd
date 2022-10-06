#!python
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from capnp.includes.schema_cpp cimport WordArray
from capnp.includes.capnp_cpp cimport DynamicStruct, DynamicStruct_Builder

cdef extern from "msgvec.h":
    cdef cppclass MsgVec:
        MsgVec(const string &jsonConfig) except +
        size_t obs_size()
        size_t act_size()

        bool input(const vector[uchar] &bytes) except +
        TimeoutResult get_obs_vector(float *obsVector) except +
        bool get_act_vector(float *actVector) except +
        vector[WordArray] get_action_command(const float *actVector) except +

cdef extern from "msgvec.h" namespace "MsgVec":
    cpdef enum TimeoutResult:
        MESSAGES_TIMED_OUT
        MESSAGES_WITHIN_TIMEOUT 