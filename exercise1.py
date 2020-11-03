from typing import Iterable
import random
import math
from dataclasses import dataclass
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Interval:
    left: float
    right: float
    label: int


def sample(n_samples: int, probabilities: Iterable[float], method: str) -> Iterable[int]:
    """Sample from discrete pd.

    Parameters
    ----------
    n_samples: number of samples to sample
    probabilities: an iterable (e.g. list) of probabilities of individual classes
    method: either `alias` or `cdf`

    Returns
    -------
    samples: samples drawn from distribution specified by probabilities argument
    """
    sampler = Sampler(probabilities, initialize=True)
    return sampler.sample(n_samples, method)


class Sampler:
    def __init__(self, probabilities: Iterable[float], initialize=False):
        """

        Parameters
        ----------
        probabilities: an iterable (e.g. list) of probabilities of individual classes
        initialize: whether to preprocess probabilities for each of the methods
        """
        if not math.isclose(np.sum(probabilities), 1):
            raise ValueError("Probabilities should sum to one!")
        self.probabilities = probabilities
        self.prob_to_class = {i + 1: prob for i, prob in enumerate(probabilities)}
        # for Alias method
        self.U = None
        self.K = None
        self.n = len(self.prob_to_class)

        self.intervals = None
        if initialize:
            self.compute_alias_tables()
            self.compute_intervals()

    def sample(self, n_samples: int, method: str = "alias") -> Iterable[int]:
        """Sample from pd.

        Parameters
        ----------
        n_samples: number of samples to sample
        method: either `alias` or `cdf`

        Returns
        -------
        samples: samples drawn from distribution specified by probabilities argument
        """
        if method == "alias":
            return self.alias_method(n_samples)
        elif method == "cdf":
            return self.bruteforce_cdf_method(n_samples)
        else:
            raise ValueError(f"Method {method} not recognized!")

    def compute_intervals(self) -> None:
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

    def compute_alias_tables(self) -> None:
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

    def alias_method(self, n_samples: int) -> Iterable[int]:
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

    def bruteforce_cdf_method(self, n_samples: int) -> Iterable[int]:
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
    if len(sys.argv) == 1 or sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print("Usage:")
        print("To createa comparison plot (saved to comparison.png):")
        print(f"\tpython {sys.argv[0]} compare <min_classes> <max_classes>")
        print("To sample with both methods and compare results using a random pd:")
        print(f"\tpython {sys.argv[0]} sample <n_classes> <n_samples>")
    elif sys.argv[1] == "compare":
        n_samples = 100
        min_classes = int(sys.argv[2])
        max_classes = int(sys.argv[3])
        alias_times = []
        cdf_times = []
        prob_vector_sampler = Sampler([1/max_classes] * max_classes, initialize=True)
        unnormalized_probs = prob_vector_sampler.sample(max_classes, method="alias")
        for n_classes in range(min_classes, max_classes + 1):
            probs = np.array(unnormalized_probs[:n_classes])
            probs = probs / np.sum(probs)
            sampler = Sampler(probs, initialize=True)
            t0 = time.time()
            sampler.sample(n_samples, "alias")
            alias_times.append((time.time() - t0) / n_samples)
            t0 = time.time()
            sampler.sample(n_samples, "cdf")
            cdf_times.append((time.time() - t0) / n_samples)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(list(range(min_classes, max_classes + 1)), alias_times, label="alias method")
        ax.plot(list(range(min_classes, max_classes + 1)), cdf_times, label="cdf method")
        ax.legend()
        fig.savefig("comparison.png")
        plt.close(fig)

    elif sys.argv[1] == "sample":
        from collections import Counter
        vector_length = int(sys.argv[2])
        n_samples = int(sys.argv[3])
        prob_vector_sampler = Sampler([1/vector_length] * vector_length, initialize=True)
        probs = np.array(prob_vector_sampler.sample(vector_length, method="alias"))
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
    else:
        print("Option not recognized, exiting.")
