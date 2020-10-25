import numpy as np
from typing import Iterable
import random
import math
from dataclasses import dataclass


@dataclass
class Interval:
    left: float
    right: float
    label: int


class Sampler:
    def __init__(self, probabilities: Iterable[float]):
        if not math.isclose(np.sum(probabilities), 1):
            raise ValueError("Probabilities should sum to one!")
        self.probabilities = probabilities
        self.prob_to_class = {i + 1: prob for i, prob in enumerate(probabilities)}
        # for Alias method
        self.U = None
        self.K = None
        self.n = len(self.prob_to_class)

        self.intervals = None

    def sample(self, n_samples, method="alias"):
        pass

    def compute_intervals(self):
        self.intervals = []
        for i, prob in sorted(
            self.prob_to_class.items(), reverse=True, key=lambda x: x[1]
        ):
            if not self.intervals:
                self.intervals.append(Interval(left=0.0, right=prob, label=i))
            else:
                self.intervals.append(
                    Interval(
                        left=self.intervals[-1].right,
                        right=self.intervals[-1].right + prob,
                        label=i,
                    )
                )

    def compute_alias_tables(self):
        self.U = [0] * (self.n + 1)
        self.K = [None] * (self.n + 1)
        overfull = set()
        underfull = set()
        exactly_full = set()
        for i, prob in self.prob_to_class.items():
            self.U[i] = self.n * prob
            if self.U[i] > 1:
                overfull.add(i)
            elif math.isclose(self.U[i], 1):
                exactly_full.add(i)
                self.K[i] = i
            else:
                underfull.add(i)
        while not len(exactly_full) == len(self.prob_to_class):
            overfull_entry = random.sample(overfull, 1)[0]
            overfull.remove(overfull_entry)
            underfull_entry = random.sample(underfull, 1)[0]
            underfull.remove(underfull_entry)
            self.K[underfull_entry] = overfull_entry
            self.U[overfull_entry] = self.U[overfull_entry] + self.U[underfull_entry] - 1
            exactly_full.add(underfull_entry)
            if math.isclose(self.U[overfull_entry], 1) or self.K[overfull_entry]:
                exactly_full.add(overfull_entry)
            elif self.U[overfull_entry] > 1:
                overfull.add(overfull_entry)
            elif self.U[overfull_entry] < 1 and not self.K[overfull_entry]:
                underfull.add(overfull_entry)
            else:
                raise ValueError("Not posible")

    def alias_method(self, n_samples):
        if not self.U:
            self.compute_alias_tables()
        samples = []
        for _ in range(n_samples):
            u = random.random()
            nu = self.n * u
            i = math.floor(nu) + 1
            y = nu + 1 - i
            if y < self.U[i]:
                samples.append(i)
            else:
                samples.append(self.K[i])
        return samples

    def bruteforce_cdf_method(self, n_samples):
        if not self.intervals:
            self.compute_intervals()
        samples = []
        for _ in range(n_samples):
            u = random.random()
            for interval in self.intervals:
                if interval.left <= u < interval.right:
                    samples.append(interval.label)
        return samples


if __name__ == "__main__":
    from collections import Counter
    vector_length = 50
    n_samples = 100000
    probs = np.array([random.randint(1, 100) for _ in range(vector_length)])
    probs = probs / np.sum(probs)
    sampler = Sampler(probs)
    print(sampler.prob_to_class)
    alias_results = Counter(sampler.alias_method(n_samples))
    cdf_results = Counter(sampler.bruteforce_cdf_method(n_samples))
    for i, probs in sampler.prob_to_class.items():
        print(f"Class {i}")
        print(f"\tProbability: {probs:.6f}")
        print(f"\tExpected count: {math.floor(n_samples * probs)}")
        print(f"\tAlias method count: {alias_results[i]}")
        print(f"\tcdf method count: {cdf_results[i]}")
