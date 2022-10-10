# distutils: language = c++

from libcpp.vector cimport vector
from cpython cimport array

from pymsgvec cimport MsgVec, TimeoutResult
from typing import List, Tuple
from enum import IntEnum, auto

from cereal import log
from capnp.includes.schema_cpp cimport WordArray, WordArrayPtr
from capnp.includes.capnp_cpp cimport DynamicStruct, DynamicStruct_Builder


cdef extern from "<utility>" namespace "std":
    vector[WordArray] move(vector[WordArray]) # Cython has no function templates

class PyTimeoutResult(IntEnum):
    MESSAGES_NOT_READY = 0
    MESSAGES_PARTIALLY_READY = auto()
    MESSAGES_ALL_READY = auto()

cdef class PyMsgVec:
    cdef MsgVec *c_msgvec

    def __cinit__(self, config_json):
        self.c_msgvec = new MsgVec(config_json)

    def __dealloc__(self):
        del self.c_msgvec

    def obs_size(self):
        return self.c_msgvec.obs_size()

    def act_size(self):
        return self.c_msgvec.act_size()

    def input(self, obs: bytes) -> bool:
        return self.c_msgvec.input(obs)

    def get_obs_vector(self) -> Tuple[PyTimeoutResult, List[float]]:
        cdef vector[float] obs_vector = vector[float](self.c_msgvec.obs_size())
        timeout_res = self.c_msgvec.get_obs_vector(obs_vector.data())
        return PyTimeoutResult(timeout_res), list(obs_vector)

    # Same as get_obs_vector but doesn't return the timeout info
    def get_obs_vector_raw(self) -> List[float]:
        timeout, obs_vector = self.get_obs_vector()
        return obs_vector

    def get_act_vector(self) -> List[float]:
        cdef vector[float] act_vector = vector[float](self.c_msgvec.act_size())
        self.c_msgvec.get_act_vector(act_vector.data())
        return list(act_vector)

    def get_reward(self) -> Tuple["bool", float]:
        cdef float reward
        cdef bool valid = self.c_msgvec.get_reward(&reward)
        return valid, reward

    def get_action_command(self, act: List[float]):
        assert len(act) == self.c_msgvec.act_size()
        cdef array.array a = array.array('f', act)

        cdef vector[WordArray] result = move(self.c_msgvec.get_action_command(a.data.as_floats))
        pyresult = []

        cdef array.array word_array_template = array.array('L', [])
        cdef array.array newarray
        for i in range(result.size()):
            newarray = array.clone(word_array_template, 0, True)
            array.extend_buffer(newarray, <char *>result[i].begin(), result[i].size())
            with log.Event.from_bytes(newarray.tobytes()) as msg:
                pyresult.append(msg)

        return pyresult
