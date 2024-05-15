import os
import pickle
from pprint import pprint
import clize
import csv

import numpy as np


def split_path(path: str):
    components = path.split('/')
    if components[-1] == '':
        return components[:-1]
    return components

def subdirs(dir):
    return (subdir for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir)))


def get_run_n_formulas(run_dir):
    eval_path = os.path.join(run_dir, 'train', 'eval.csv')
    if not os.path.isfile(eval_path):
        return None
    with open(eval_path, 'r') as file:
        rows = tuple(csv.reader(file))
    assert rows[0] == ['split', 'return', 'n_formulas'], f'Invalid header: {rows[0]} (run_dir: {run_dir})'
    rows = rows[1:]
    return rows[0][2]


def get_run_steps(run_dir):
    curriculum_path = os.path.join(run_dir, 'train', 'curriculum.csv')
    if not os.path.isfile(curriculum_path):
        return None
    with open(curriculum_path, 'r') as file:
        rows = tuple(csv.reader(file))
    assert rows[0] == ['Wall time', 'Step', 'Value'], f'Invalid header: {rows[0]} (run_dir: {run_dir})'
    return int(rows[-1][1])


def get_run_eval(run_dir, gamma):
    with open(os.path.join(run_dir, 'train', 'eval.pkl'), 'rb') as file:
        raw_eval = pickle.load(file)
    returns = raw_eval['returns']
    steps = raw_eval['steps']
    eval = {}
    for split in returns.keys():
        rets = np.array(returns[split])
        stps = np.array(steps[split])
        eval[split] = dict(
            return_mean=rets.mean(),
            steps_mean=stps.mean(),
            discounted_return_mean=(gamma**stps * rets).mean(),
        )
    return eval


def get_runs_eval(runs_dir, gamma):
    n_formulas = []
    n_steps = []
    returns = {}
    ep_lengths = {}
    discounted_returns = {}
    for run_dir in subdirs(runs_dir):
        run_n_formulas = get_run_n_formulas(os.path.join(runs_dir, run_dir))
        if run_n_formulas == 'None' or run_n_formulas is None:
            print(f'A run from {run_dir} has missing n_formulas')
            return None
        n_formulas.append(int(run_n_formulas))
        n_steps.append(get_run_steps(os.path.join(runs_dir, run_dir)))
        run_eval = get_run_eval(os.path.join(runs_dir, run_dir), gamma=gamma)
        for split, ret in run_eval.items():
            returns.setdefault(split, []).append(ret['return_mean'])
            ep_lengths.setdefault(split, []).append(ret['steps_mean'])
            discounted_returns.setdefault(split, []).append(ret['discounted_return_mean'])
    flat_eval = {}
    flat_eval[f'n_formulas_mean'] = np.mean(n_formulas)
    flat_eval[f'n_formulas_std'] = np.std(n_formulas)
    flat_eval[f'n_formulas_min'] = np.min(n_formulas)
    flat_eval[f'n_formulas_max'] = np.max(n_formulas)
    flat_eval[f'n_steps_mean'] = np.mean(n_steps)
    flat_eval[f'n_steps_std'] = np.std(n_steps)
    for split in returns.keys():
        flat_eval[f'{split}_return_mean'] = np.mean(returns[split])
        flat_eval[f'{split}_return_std'] = np.std(returns[split])
        flat_eval[f'{split}_ep_length_mean'] = np.mean(ep_lengths[split])
        flat_eval[f'{split}_ep_length_std'] = np.std(ep_lengths[split])
        flat_eval[f'{split}_discounted_return_mean'] = np.mean(discounted_returns[split])
        flat_eval[f'{split}_discounted_return_std'] = np.std(discounted_returns[split])
    return flat_eval


def main(topdir, *, gamma=0.94):
    evals = { runs_dir: get_runs_eval(os.path.join(topdir, runs_dir), gamma=gamma) for runs_dir in subdirs(topdir)}
    skipped = []
    for k in evals.keys():
        if evals[k] == None:
            print(f'Skipped subdir {k} (missing evals)')
            skipped.append(k)
    for k in skipped:
        del evals[k]
    keys = tuple(next(iter(evals.values())).keys())
    print(keys)
    pprint(evals)
    assert all(set(eval.keys()) == set(keys) for eval in evals.values())
    with open(os.path.join(topdir, 'eval.csv'), 'w') as file:
        writer = csv.writer(file)
        header = ['config'] + list(keys)
        writer.writerow(header)
        for runs_dir, eval in evals.items():
            row = [runs_dir] + [eval[key] for key in keys]
            writer.writerow(row)
    with open(os.path.join(topdir, 'eval.pkl'), 'wb') as file:
        pickle.dump(evals, file)


if __name__ == '__main__':
    clize.run(main)

