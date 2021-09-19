import copy
import math
from typing import List
from typing import Tuple

import numpy as np
from numpy import ndarray


class Problem:

    def __init__(self, A: ndarray, b: ndarray, c: ndarray):
        # TODO: exception
        self._A: ndarray = A
        self._b: ndarray = b
        self._c: ndarray = c


    @property
    def A(self):
        return copy.deepcopy(self._A)


    @property
    def b(self):
        return copy.deepcopy(self._b)


    @property
    def c(self):
        return copy.deepcopy(self._c)


    def split_problem(self, x: ndarray, i: int) -> Tuple['Problem', 'Problem']:
        """
        :param x: a components list;
        :param i: an index which indicates a component to be a spliterator;
        """
        lower_bound = math.floor(x[i])
        greater_bound = math.ceil(x[i])

        constraint_dimension = self._A.shape[1]
        le_than_bound_constraint = [0 for _ in range(constraint_dimension)]  # less or equals
        ge_than_bound_constraint = [0 for _ in range(constraint_dimension)]  # greater or equals
        le_than_bound_constraint[i] = 1
        ge_than_bound_constraint[i] = -1

        return (copy.deepcopy(self).append_constraint(le_than_bound_constraint, lower_bound),
                copy.deepcopy(self).append_constraint(ge_than_bound_constraint,
                                                      (-1) * greater_bound))


    def append_constraint(self, A_row: List[int], b_row: int) -> 'Problem':
        self._A = np.vstack([self._A, A_row])
        self._b = np.hstack([self._b, [b_row]])
        return self
