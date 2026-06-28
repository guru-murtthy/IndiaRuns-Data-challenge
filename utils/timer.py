import time
from utils.logger import log_info

class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        log_info(f"{self.name} completed in {elapsed:.3f} seconds.")
