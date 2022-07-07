import tempfile

from contextlib import contextmanager
from cereal import log


@contextmanager
def artificial_logfile():
    f = tempfile.NamedTemporaryFile("wb+")
    event = log.Event.new_message()
    event.write(f)
    f.seek(0)
    yield f
    f.close()
