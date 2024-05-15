import random
import numpy
import torch
import collections
import os
import numpy as np


def set_seed(seed: int):
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def average_reward_per_step(returns, num_frames):
    avgs = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        avgs.append(returns[i] / num_frames[i])

    return numpy.mean(avgs)


def average_discounted_return(returns, num_frames, disc):
    discounted_returns = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        discounted_returns.append(returns[i] * (disc ** (num_frames[i]-1)))

    return numpy.mean(discounted_returns)