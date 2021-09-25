from ast import literal_eval
from typing import Tuple

import numpy as np


class SetupReader:
    #
    _BASE_DIR = 'setup/'


    @staticmethod
    def read_setup(filename: str, subdirectory: str = ''):
        lines = SetupReader._read_file(filename, subdirectory)
        return SetupReader._map_str_to_dict(lines)


    @staticmethod
    def _read_file(filename: str, subdirectory: str = '') -> Tuple[str]:
        lines: tuple
        path = SetupReader._BASE_DIR + subdirectory + '/' + filename
        with open(path, encoding = 'utf-8') as file:
            lines = tuple(line.strip() for line in file if line.isspace() is not True)

        return lines


    @staticmethod
    def _map_str_to_dict(lines: Tuple[str]) -> dict:
        kvalues = dict()
        for line in lines:
            key, value = line.split('=')
            value = literal_eval(value.replace('"', '').replace(' ', ''))
            kvalues.update({key: np.array(value)})

        return kvalues
