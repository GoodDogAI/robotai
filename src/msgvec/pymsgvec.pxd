#!python
#cython: language_level=3

from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint16_t, uint32_t

from capnp.includes.schema_cpp cimport WordArray
from capnp.includes.capnp_cpp cimport DynamicStruct as C_DynamicStruct, DynamicStruct_Builder

cdef extern from "msgvec.h":
    cdef cppclass CerealEventReader "cereal::Event::Reader"

    cdef cppclass MsgVec:
        MsgVec(const string &jsonConfig, const MessageTimingMode timingMode) except +
        unordered_set[string] get_possible_event_types() except +
        size_t obs_size()
        size_t act_size()
        size_t vision_size()
        bool is_discrete_act()

        InputResult input(const C_DynamicStruct.Reader &evt) except +
        InputResult input(const vector[uchar] &bytes) except +

        void input_vision(const float *visionIntermediate, const uint32_t frameId) except +
        TimeoutResult get_obs_vector(float *obsVector) except +
        bool get_act_vector(float *actVector) except +
        bool get_reward(float *reward) except +
        vector[WordArray] get_action_command(const float *actVector) except +

        void _debug_print_timing()

cdef extern from "msgvec.h" namespace "MsgVec":
    cpdef enum MessageTimingMode:
        REALTIME"MsgVec::MessageTimingMode::REALTIME"
        REPLAY"MsgVec::MessageTimingMode::REPLAY"

    cpdef enum TimeoutResult:
        MESSAGES_NOT_READY
        MESSAGES_PARTIALLY_READY
        MESSAGES_ALL_READY

    cdef struct InputResult:
        bool msg_processed
        bool act_ready

 