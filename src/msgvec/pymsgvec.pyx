# distutils: language = c++

from pymsgvec cimport MsgVec

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
