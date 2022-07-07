import tempfile

from contextlib import contextmanager
from cereal import log


@contextmanager
def artificial_logfile(count: int = 1, video: bool = False):
    f = tempfile.NamedTemporaryFile("wb+", suffix=".log")

    assert count >= 1
    for i in range(count):
        event = log.Event.new_message()

        if video:
            event.init("headEncodeData")
            # \xd0 will test that the base64 encoding, since it's not a valid unicode string
            event.headEncodeData.data = b"\x00\x01\x02\x03\xd0"

        event.write(f)

    f.seek(0)
    yield f
    f.close()
