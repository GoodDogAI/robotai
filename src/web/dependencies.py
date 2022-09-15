from src.config import HOST_CONFIG
from src.logutil import LogHashes

_loghashes = None

def get_loghashes() -> LogHashes:
    global _loghashes
    if _loghashes is None:
        _loghashes = LogHashes(HOST_CONFIG.RECORD_DIR)
    return _loghashes
