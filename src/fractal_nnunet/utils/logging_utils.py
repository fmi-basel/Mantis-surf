# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#
"""Helper functions for capturing print statements in dependencies and redirecting to logger output."""

import logging
import sys
from contextlib import contextmanager
from io import StringIO


# Custom stream that logs instead of printing
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = StringIO()

    def write(self, message):
        for line in message.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass  # Required by file-like interface


# Context manager to redirect stdout/stderr
@contextmanager
def redirect_print_to_logger(logger):
    stdout_original = sys.stdout
    stderr_original = sys.stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout = stdout_original
        sys.stderr = stderr_original
