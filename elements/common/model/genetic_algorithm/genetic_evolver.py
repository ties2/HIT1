import random
from typing import Dict, Tuple, Callable, Any

import torch
import numpy as np
import numpy.random as npr


class Evolver:
    """    Single offspring genetic algorithm for deep learning hyperparameter optimisation."""
    def __init__(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, meta: Dict[str, Tuple[float, float, float]], fitness_f: Callable[[np.ndarray, Any], float], writer, classes: list, num_gens=100, pop: np.ndarray = None, nr_best=5, mut_prob=0.8, sigma=0.2, zero_part=0.001, max_tries=100, sign_digits=5, working_dir=None):
        """
        Initialise Evolver.
        :param meta: The dictionary with evolution metadata with for each parameter: name_parameter, (gain, lower_limit, upper_limit).
        :param fitness_f: The fitness function.
        :param num_gens: The number of generations to evolve.
        :param pop: The initial population for the hyperparameters
        :param nr_best: The number of best parents that are use to generate offspring.
        :param mut_prob: The mutation probability.
        :param sigma: The sigma for the Gaussian distribution used for the mutation.
        :param zero_part: If parameter optimize to zero, mutations are generated in range zero_part * total range of parameter.
        :param max_tries: The number of attempts to generate a unique mutation.
        :param sign_digits: The number of significant digits for a hyperparameter.
        :param tb: The TB writer if any.
        """
        self.meta = meta
        self.num_params = len(self.meta)
        self.fitness_f = fitness_f
        self.num_gens = num_gens
        self.pop = pop if pop else np.ndarray((0, self.num_params+1))
        assert self.pop.shape[1] == self.num_params+1, f'2nd dimension of population should be {self.num_params + 1}, = num_params + 1, last index is for fitness '
        self.nr_best = nr_best
        self.mut_prob = mut_prob
        self.sigma = sigma
        self.zero_part = zero_part
        self.max_tries = max_tries
        self.sign_digits = sign_digits
        self.gain = np.array([self.meta[k][0] for k in self.meta.keys()])
        self.low_limit = np.array([self.meta[k][1] for k in self.meta.keys()])
        self.high_limit = np.array([self.meta[k][2] for k in self.meta.keys()])
        self.delta_zero = self.zero_part * np.array([max(abs(self.meta[k][1]), abs(self.meta[k][2])) for k in self.meta.keys()])
        self._sort()
        self.tb = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.all_generations = list()
        self.working_dir = working_dir
        self.classes = classes

    def _sort(self):
        """Sort the population"""
        self.pop = self.pop[np.argsort(-self.pop[:, -1])]               # sort in descending order

    def child(self) -> np.ndarray:
        """Generate offspring"""
        if not len(self.pop):
            return (self.low_limit + self.high_limit) / 2    # first child
        n = min(self.nr_best, len(self.pop))                            # select best n as parents
        w = self.pop[:n, -1] - self.pop[:n, -1].min() + 1E-6           # reverse fitness as weights (sum > 0)
        res = self.pop[random.choices(range(n), weights=w)[0]][:-1]     # weighted crossover selection, remove fitness
        for t in range(self.max_tries):                                 # mutate until a change occurs (prevent duplicates)
            # gain = zero, no changes, hyper
            # npr.randn, gaussian distribution of sigma + deviation
            # +1 to allow for changes in the multiplication
            # delta prevents getting stick in zeros
            scale = (self.gain * (npr.random(self.num_params) < self.mut_prob) * npr.randn(self.num_params) * npr.random() * self.sigma + 1)
            delta = np.ones(self.num_params) * (res == 0) * (npr.random(self.num_params) < self.mut_prob) * npr.randn(self.num_params) * self.delta_zero  # mutate zeros
            res = res * scale + delta                                   # mutate
            res = np.round(np.clip(res, self.low_limit, self.high_limit), self.sign_digits)  # clip
            # check if new child is different from before
            if not np.any(np.all(self.pop[:, :-1] == res, axis=1)): break
        return res

    def add(self, child: np.ndarray, fitness: float):
        """Add child with his fitness to population"""
        self.pop = np.vstack((self.pop, np.append(child, fitness)))
        self._sort()

    def best(self) -> Tuple[np.ndarray, float]:
        """Get the best set of hyperparameters and its fitness value"""
        return self.pop[0][:-1], self.pop[0][-1]

    def run(self, verbose=False) -> Tuple[np.ndarray, float]:
        """Run the evolver and return the best set of hyperparameters and its fitness value"""

        for g in range(self.num_gens):
            c = self.child()
            f = self.fitness_f(c, self,  train_loader=self.train_loader, test_loader=self.test_loader, classes=self.classes, gen_number=g, working_dir=self.working_dir)
            self.add(c, f)
            if verbose: print(f'gen={g}, child={c},{f}, best={self.best()}')
        return self.best()


