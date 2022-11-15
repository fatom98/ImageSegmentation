from typing import Callable

from pyswarm import pso


class ParticleSwarm:
    def __init__(self,
                 fitness_function: Callable,
                 upper_bound: list[int],
                 lower_bound: list[int],
                 swarm_size: int = 20,
                 max_iteration: int = 10):
        self.__fitness_function: Callable = fitness_function
        self.__upper_bound: list[int] = upper_bound
        self.__lower_bound: list[int] = lower_bound
        self.__swarm_size: int = swarm_size
        self.__max_iteration: int = max_iteration

        self.__global_best_k: int = 0
        self.__global_best_iteration: int = 0
        self.__global_best_attempts: int = 0
        self.__global_best_epsilon: float = 0.0
        self.__objective_value_at_best: int = 0

    def run(self) -> None:
        print("Finding optimum parameters")

        ret = pso(func=self.__fitness_function,
                  lb=self.__lower_bound,
                  ub=self.__upper_bound,
                  swarmsize=self.__swarm_size,
                  maxiter=self.__max_iteration)

        self.__assign_global_best_values(ret)

    @property
    def result(self):
        return f"Best K: {self.__global_best_k}, Best iteration: {self.__global_best_iteration}, " \
               f"Best attempts: {self.__global_best_attempts}, Best epsilon: {self.__global_best_epsilon}," \
               f" Best objective value: {self.__objective_value_at_best:_}"

    def __assign_global_best_values(self, ret):
        self.__global_best_k = round(ret[0][0])
        self.__global_best_iteration = round(ret[0][1])
        self.__global_best_attempts = round(ret[0][2])
        self.__global_best_epsilon = round(ret[0][3], 2)
        self.__objective_value_at_best = ret[1]
