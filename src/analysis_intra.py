import os
import pickle
from pprint import pprint
import clize
from matplotlib import pyplot as plt
import numpy as np
import yaml
from dataset import load_dataset


# std/mean
def rstd(xs, axis=None):
    if axis is None:
        return np.std(xs) / np.mean(xs)
    return np.std(xs, axis=axis) / np.mean(xs, axis=axis)

def get_variables(spec: str, load_test: bool):
    dataset_name, kernel_name = spec.split('-')
    print(f'Dataset: {dataset_name}, kernel: {kernel_name}')
    # config, automata, _, _, kernel_reprs
    ds = load_dataset(os.path.join('datasets', dataset_name), kernel=kernel_name)
    if load_test:
        ds_test = load_dataset(os.path.join('datasets', f'{dataset_name}test'), kernel=kernel_name)
        ds4 = ds[4] + ds_test[4]
    else:
        ds4 = ds[4]
    intras = np.array([rstd(k, axis=0).mean() for k in ds4])
    n_states = [k.shape[0] for k in ds4]

    variables = {
        'intras': intras,
        'n_states': n_states,
    }

    return variables


def main(*specs, x: str = None, y: str = None, alpha: float = 0.7, log: bool = False, bins: int = 10, exclude_test: bool = False):
    if x is None and y is None:
        x, y = 'n_states', None

    variables = { spec: get_variables(spec, exclude_test) for spec in specs }

    if y is not None:
        for k, vars in variables.items():
            plt.scatter(vars[x], vars[y], s=2, label=k, alpha=alpha)
    else:
        for k, vars in variables.items():
            plt.hist(vars[x], bins=bins, label=k, alpha=alpha)

    titles = {
        'intras': 'Mean RSD of kernel representations',
        'n_states': 'Number of states in the DFA',
    }

    plt.xlabel(titles[x])
    if y is not None:
        plt.ylabel(titles[y])

    if len(specs) > 1:
        plt.legend()

    if log:
        plt.yscale('log')

    plt.show()


if __name__ == '__main__':
    clize.run(main)
