# distutils: language = c++

from libcpp.vector cimport vector

from pymsgvec cimport MsgVec
from typing import List

cdef class PyMsgVec:
    cdef MsgVec *c_msgvec

    def __cinit__(self, config_json):
        self.c_msgvec = new MsgVec(config_json)

    def __dealloc__(self):
        del self.c_msgvec

    def obs_size(self):
        return self.c_msgvec.obs_size()

    def input(self, obs: bytes) -> bool:
        return self.c_msgvec.input(obs)

    def get_obs_vector(self) -> List[float]:
        cdef vector[float] obs_vector = vector[float](self.c_msgvec.obs_size())
        self.c_msgvec.get_obs_vector(obs_vector.data())
        return list(obs_vector)