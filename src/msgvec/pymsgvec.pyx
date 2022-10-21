# distutils: language = c++
import json

import numpy as np
cimport numpy as cnp

from libcpp.vector cimport vector
from cpython cimport array

from pymsgvec cimport MsgVec, TimeoutResult, MessageTimingMode
from typing import List, Tuple, Dict, Set, Union
from enum import IntEnum, auto

from cereal import log

from capnp.includes.schema_cpp cimport WordArray, WordArrayPtr
from capnp.includes.capnp_cpp cimport DynamicStruct as C_DynamicStruct, DynamicStruct_Builder
from capnp.lib.capnp cimport _DynamicStructReader, _DynamicStructBuilder

cdef extern from "<utility>" namespace "std":
    vector[WordArray] move(vector[WordArray]) # Cython has no function templates

class PyTimeoutResult(IntEnum):
    MESSAGES_NOT_READY = 0
    MESSAGES_PARTIALLY_READY = auto()
    MESSAGES_ALL_READY = auto()

class PyMessageTimingMode(IntEnum):
    REALTIME = 0
    REPLAY = auto()


cdef class PyMsgVec:
    cdef MsgVec *c_msgvec
    config_dict: Dict
    config_json: bytes
 
    def __cinit__(self, config_dict: Dict, timing_mode: PyMessageTimingMode):
        self.config_dict = config_dict
        self.config_json = json.dumps(config_dict).encode("utf-8")
        self.c_msgvec = new MsgVec(self.config_json, <MessageTimingMode><int>timing_mode)

    def __dealloc__(self):
        del self.c_msgvec

    def obs_size(self):
        return self.c_msgvec.obs_size()

    def act_size(self):
        return self.c_msgvec.act_size()

    def input(self, message):
        cdef _DynamicStructReader reader
        cdef C_DynamicStruct.Reader c_reader
    
        if isinstance(message, _DynamicStructReader):
            reader = message
            c_reader = <C_DynamicStruct.Reader>reader.thisptr 
        elif isinstance(message, _DynamicStructBuilder):
            reader = message.as_reader()
            c_reader = <C_DynamicStruct.Reader>reader.thisptr 
        else:
            raise TypeError("message must be a DynamicStructReader or DynamicStructBuilder")

        return self.c_msgvec.input(c_reader)

    def input_vision(self, vision_vector: np.ndarray[float], frame_id: int):
        assert vision_vector.shape == (self.c_msgvec.vision_size(),)
        assert vision_vector.dtype == np.float32
        # From https://stackoverflow.com/questions/10718699/convert-numpy-array-to-cython-pointer
        cdef const cnp.float32_t[::1] vision_buf = np.ascontiguousarray(vision_vector, dtype=np.float32)
        self.c_msgvec.input_vision(&vision_buf[0], frame_id)

    def get_obs_vector(self) -> Tuple[PyTimeoutResult, np.ndarray[float]]:
        cdef vector[float] obs_vector = vector[float](self.c_msgvec.obs_size())
        timeout_res = self.c_msgvec.get_obs_vector(obs_vector.data())
        return PyTimeoutResult(timeout_res), np.array(obs_vector, dtype=np.float32)

    # Same as get_obs_vector but doesn't return the timeout info
    def get_obs_vector_raw(self) -> np.ndarray[float]:
        timeout, obs_vector = self.get_obs_vector()
        return obs_vector

    def get_act_vector(self) -> np.ndarray[float]:
        cdef vector[float] act_vector = vector[float](self.c_msgvec.act_size())
        self.c_msgvec.get_act_vector(act_vector.data())
        return np.asarray(act_vector, dtype=np.float32)

    def get_reward(self) -> Tuple["bool", float]:
        cdef float reward
        cdef bool valid = self.c_msgvec.get_reward(&reward)
        return valid, reward

    def get_action_command(self, act: np.ndarray[float]):
        assert act.shape == (self.c_msgvec.act_size(),)
        assert act.dtype == np.float32
        cdef const cnp.float32_t[::1] act_buf = np.ascontiguousarray(act, dtype=np.float32)

        # Special case if action size is zero (no outputs, useful for tests), we want to have something to index on, so we just expand the size
        if self.c_msgvec.act_size() == 0:
            act_buf = np.ascontiguousarray([0.0], dtype=np.float32)

        cdef vector[WordArray] result = move(self.c_msgvec.get_action_command(&act_buf[0]))
        pyresult = []

        cdef array.array word_array_template = array.array('L', [])
        cdef array.array newarray
        for i in range(result.size()):
            newarray = array.clone(word_array_template, 0, True)
            array.extend_buffer(newarray, <char *>result[i].begin(), result[i].size())
            with log.Event.from_bytes(newarray.tobytes()) as msg:
                pyresult.append(msg)

        return pyresult
