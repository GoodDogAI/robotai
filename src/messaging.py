import time
import capnp
from typing import Optional
from cereal import log


# Note, this function is very similar to one provided by cereal, but it's logMonoTime will be synced
# with the time for other realtime messages
def new_message(service: Optional[str] = None, size: Optional[int] = None) -> capnp.lib.capnp._DynamicStructBuilder:
  dat = log.Event.new_message()
  dat.logMonoTime = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
  dat.valid = True
  if service is not None:
    if size is None:
      dat.init(service)
    else:
      dat.init(service, size)
  return dat