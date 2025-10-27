import os
from pathlib import Path
import logging


class Logger:
    def __init__(self, _name, _root_dir, level=logging.DEBUG, ) -> None:
        self._root_dir = Path(_root_dir)
        os.makedirs(self._root_dir, exist_ok=True)
        self.logger = logging.getLogger(_name)
        self.logger.setLevel(level)

    def addFileHandler(self,
                       fname: str,
                       level=None,
                       _format="[%(asctime)s, %(levelname)s] : %(message)s"):
        handler = logging.FileHandler(self._root_dir / fname, 'w', encoding = 'utf-8')

        formatter = logging.Formatter(_format)

        if level is not None:
            handler.setLevel(level)
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def debug(self, line):
        self.logger.debug(line)

    def info(self, line):
        self.logger.info(line)
