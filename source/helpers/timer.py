import time
import logging

class Timer:
    def __init__(self, type: str) -> None:
        self.type = type
        self.start_time = time.time()

    def stop(self) -> None:
        end_time = time.time()
        logger = logging.getLogger("main")
        logger.info(f"{self.type} took {end_time - self.start_time} seconds")