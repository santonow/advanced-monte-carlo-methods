# Advanced Monte Carlo methods
This is a repository containing my assignments for Advanced Monte Carlo Methods course.

## Exercise 1

Write a sampler that samples from a discrete probability distribution based on:
- cumulative distribution function
- alias method

### Usage:

To create `comparison.png` plot:
```bash
python exercise1.py compare <min_classes> <max_classes>
```

For example `python exercise1.py compare 1 10000` will create a following plot:

![comparison.png](comparison.png)

It will take roughly 18 seconds to create this plot.

Looking at this plot it is clear that cdf method takes linear time relative to the
number of classes, whereas Alias method takes constant (O(1)) time relative to the number of classes
(the length of probability vector).

To create a comparison between each method in terms of their validity
(i.e. if the method's results are roughly the same and consistent with the
probability vector specified) use the following script call:
```bash
python exercise1.py sample <n_classes> <n_samples>
```

For example following call: `python exercise1.py sample 4 10000` will output this:
```python
Class 1
        Probability: 0.400000
        Expected count: 4000
        Alias method count: 4045
        cdf method count: 4040
Class 2
        Probability: 0.300000
        Expected count: 3000
        Alias method count: 2947
        cdf method count: 2979
Class 3
        Probability: 0.200000
        Expected count: 2000
        Alias method count: 1989
        cdf method count: 2015
Class 4
        Probability: 0.100000
        Expected count: 1000
        Alias method count: 1019
        cdf method count: 966
```

Usage as a package:
```python
from exercise1 import Sampler, sample
import numpy as np

# using a class
sampler = Sampler(probabilities=[0.2, 0.2, 0.3, 0.1])
print(sampler.sample(n_samples=10, method="alias"))
print(sampler.sample(n_samples=10, method="cdf"))

# using a function

print(sample(n_samples=10, probabilities=[0.2, 0.2, 0.3, 0.1], method="alias"))
print(sample(n_samples=10, probabilities=[0.2, 0.2, 0.3, 0.1], method="cdf"))

# creating random probability vectors using this class (used in `compare` and `sample` scripts):

vector_length = 100
# just specifying a uniform distribution
prob_vector_sampler = Sampler([1/vector_length] * vector_length, initialize=True)
probs = np.array(prob_vector_sampler.sample(vector_length, method="alias"))
# the vector sums to one now, making it a valid pdf
probs = probs / np.sum(probs)
```

