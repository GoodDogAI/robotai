#pragma once

// From Openpilot
#define checked_ioctl(x,y,z) { int _ret = HANDLE_EINTR(ioctl(x,y,z)); if (_ret!=0) { std::cerr << "checked_ioctl failed " << x << " " << y << " " << z << " = " << _ret << std::endl; } assert(_ret==0); }
#define checked_v4l2_ioctl(x,y,z) { int _ret = HANDLE_EINTR(v4l2_ioctl(x,y,z)); if (_ret!=0) { std::cerr << "checked_v4l2_ioctl failed " << x << " " << y << " " << z << " = " << _ret << std::endl; } assert(_ret==0); }

// keep trying if x gets interrupted by a signal
#define HANDLE_EINTR(x)                                        \
  ({                                                           \
    decltype(x) ret_;                                          \
    int try_cnt = 0;                                           \
    do {                                                       \
      ret_ = (x);                                              \
    } while (ret_ == -1 && errno == EINTR && try_cnt++ < 100); \
    ret_;                                                       \
  })
