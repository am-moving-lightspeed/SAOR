import math
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from numpy import ndarray
from scipy.optimize import linprog

from ..model.Problem import Problem
from ..model.ProblemDefinitionException import ProblemDefinitionException
from ..model.StopIterationException import StopIterationException
from ..model.StopMethodException import StopMethodException


class BranchAndBoundMethod:

    def __init__(self, problem: Problem = None, precision: int = 4):

        if not isinstance(problem, Problem) or problem is None:
            raise ProblemDefinitionException

        self._stack: List[Problem] = [problem]
        self._x_best: ndarray
        self._record: Union[int, float] = -math.inf
        self._precision: int = precision


    def invoke(self) -> ndarray:
        try:
            self._start()
        except StopMethodException:
            return self._x_best


    def _get_next_problem(self) -> Problem:
        try:
            return self._stack.pop(-1)
        except IndexError as error:
            raise StopMethodException from error


    def _push_problems(self, *args: Problem):
        self._stack += args


    @staticmethod
    def _calculate_objective_func(c: ndarray, x: ndarray) -> float:
        return np.array([c_val * x_val for c_val, x_val in zip(c, x)]).sum()


    def _check_local_result_and_update_state(self, x_local: ndarray, record_local: int) -> None:
        if record_local <= self._record:
            raise StopIterationException

        rounding_mapper = map(lambda value: round(value % 1, self._precision), x_local)
        is_integer = np.array([value for value in rounding_mapper]).sum() == 0.0

        if is_integer:
            self._record = record_local
            self._x_best = x_local.astype(int)
            raise StopIterationException


    @staticmethod
    def _find_first_not_integer(x: ndarray) -> int:
        for i, value in enumerate(x):
            if value % 1 != 0.0:
                return i
        raise StopIterationException  # should be unreachable


    @staticmethod
    def _solve_with_simplex_method(problem: Problem) -> Tuple[ndarray, int]:
        c_inverted = problem.c * -1
        result = linprog(c_inverted, problem.A, problem.b, method = 'simplex')
        return result.x, result.status


    def _start(self) -> None:
        while True:
            try:
                problem: Problem = self._get_next_problem()

                x_local, state = self._solve_with_simplex_method(problem)
                if state != 0:
                    raise StopIterationException(f'Simplex method returned an error (code {state})')

                record_local = math.floor(self._calculate_objective_func(problem.c, x_local))
                self._check_local_result_and_update_state(x_local, record_local)

                index = self._find_first_not_integer(x_local)
                new_problem_1, new_problem_2 = problem.split_problem(x_local, index)
                self._push_problems(new_problem_2, new_problem_1)

            except StopIterationException:
                pass
