# distutils: language = c++

from libcpp.vector cimport vector
from cpython cimport array

from pymsgvec cimport MsgVec
from typing import List

from capnp.includes.schema_cpp cimport WordArray
from capnp.includes.capnp_cpp cimport DynamicStruct, DynamicStruct_Builder

from capnp.lib.capnp cimport _DynamicStructReader, _DynamicStructBuilder


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

    def get_obs_vector(self) -> List[float]:
        cdef vector[float] obs_vector = vector[float](self.c_msgvec.obs_size())
        self.c_msgvec.get_obs_vector(obs_vector.data())
        return list(obs_vector)

    def get_action_command(self, act: List[float]):
        assert len(act) == self.c_msgvec.act_size()
        cdef array.array a = array.array('f', act)

        cdef vector[WordArray] result = self.c_msgvec.get_action_command(a.data.as_floats)
        pyresult = []

        return pyresult
