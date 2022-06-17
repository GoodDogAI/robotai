#pragma once

#include <atomic>
#include <csignal>
#include <cassert>
#include <thread>

// From Openpilot
// https://github.com/commaai/openpilot/blob/master/common/util.h

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

class ExitHandler {
public:
  ExitHandler() {
    std::signal(SIGINT, (sighandler_t)set_do_exit);
    std::signal(SIGTERM, (sighandler_t)set_do_exit);

#ifndef __APPLE__
    std::signal(SIGPWR, (sighandler_t)set_do_exit);
#endif
  };
  inline static std::atomic<bool> power_failure = false;
  inline static std::atomic<int> signal = 0;
  inline operator bool() { return do_exit; }
  inline ExitHandler& operator=(bool v) {
    signal = 0;
    do_exit = v;
    return *this;
  }
private:
  static void set_do_exit(int sig) {
#ifndef __APPLE__
    power_failure = (sig == SIGPWR);
#endif
    signal = sig;
    do_exit = true;
  }
  inline static std::atomic<bool> do_exit = false;
};
