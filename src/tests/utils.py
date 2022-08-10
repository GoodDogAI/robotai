import tempfile
import hashlib
import numpy as np

from contextlib import contextmanager
from cereal import log
from typing import Tuple

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

    # Write the sha256 hash for ease of testing
    f.seek(0)
    sha256_hash = hashlib.sha256()
    sha256_hash.update(f.read())
    f.sha256 = sha256_hash.hexdigest()

    f.seek(0)
    yield f
    f.close()

def get_test_image(color: Tuple[int, int, int], width: int, height: int) -> np.ndarray:
    img = np.zeros(shape=(height, width * 3), dtype=np.uint8)
    img[:, 0::3] = color[0]
    img[:, 1::3] = color[1]
    img[:, 2::3] = color[2]

    return img