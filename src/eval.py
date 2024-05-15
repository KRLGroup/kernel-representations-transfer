import csv
import os
import pickle
from pprint import pprint

import numpy as np
import clize

from dataset import load_dataset
import utils


def split_path(path: str):
    components = path.split('/')
    if components[-1] == '':
        return components[:-1]
    return components

def get_arg(model_name: str, i: int):
    # possible prefixes of pos args plus the prefix of
    # the first keyword arg
    prefixes = [
        ['Dataset'],
        ['ToyMC'],
        ['_seed:'],
    ]
    if i >= len(prefixes)-1:
        raise ValueError(f'No {i}-th arg in {model_name}')
    this_prefixes = prefixes[i]
    next_prefixes = prefixes[i+1]
    for prefix in this_prefixes:
        if prefix in model_name:
            _, tail = model_name.split(prefix)
            for next_prefix in next_prefixes:
                if next_prefix in tail:
                    tail = tail.split(next_prefix)[0]
                    break
            else:
                raise ValueError(f'No {i+1}-th arg in {model_name}')
            if tail.endswith('_'):
                tail = tail[:-1]
            return prefix + tail

def get_keyword_arg(model_name: str, key: str, next_key: str):
    _, tail = model_name.split(key + ':')
    if f'_{next_key}:' in tail:
        tail = tail.split(f'_{next_key}:')[0]
    return tail


def clean_dataset_spec(dataset_spec: str):
    if dataset_spec.endswith('_noshuffle'):
        dataset_spec = dataset_spec[:-len('_noshuffle')]
    if dataset_spec.endswith('_test'):
        dataset_spec = dataset_spec[:-len('_test')]
    if dataset_spec.endswith('_curriculum'):
        dataset_spec = dataset_spec[:-len('_curriculum')]
    return dataset_spec


def parse_model_name(model_name: str) -> dict:
    # print(f'Model name: {model_name}')
    sampler = get_arg(model_name, 0)

    env = get_arg(model_name, 1)
    assert env == 'ToyMC-v0'

    progression_mode = get_keyword_arg(model_name, 'prog', 'mlp_kernel_encoder')
    # assert progression_mode == 'kernel'

    return dict(sampler=sampler, env=env, progression_mode=progression_mode)

def parse_dataset_sampler(sampler):
    assert sampler.startswith('Dataset_'), f'sampler: {sampler}'
    # print(f'Sampler: {sampler}')
    dataset_spec = clean_dataset_spec(sampler[len('Dataset_'):])
    dataset_name, *tail = dataset_spec.split('_')
    kernel = '_'.join(tail)
    return dataset_name, kernel


def safenext(it, default=None):
    try:
        return next(it)
    except StopIteration:
        return default
    # 15_940_000


def get_n_formulas(n_frames, model_dir):
    if not os.path.isfile(os.path.join(model_dir, 'train', 'curriculum.csv')):
        return None
    with open(os.path.join(model_dir, 'train', 'curriculum.csv'), 'r') as f:
        rows = tuple(csv.reader(f))
    assert tuple(rows[0]) == ('Wall time','Step','Value'), f'Invalid header: {rows[0]}'
    n_formulas = safenext(int(float(row[2])) for row in rows[1:] if int(row[1]) == n_frames)
    return n_formulas


def do_eval(env, model_name, model_dir, sampler, progression_mode, device):
    eval = utils.Eval(
        env,
        os.path.join(*split_path(model_dir)[1:]), # remove 'storage' component
        sampler,
        seed=0, # does not have any effect if env and policy are deterministic
        device=device,
        num_procs=1,
        progression_mode=progression_mode,
        legacy_kernel_encoder=False,
        render=False
    )

    print(f'Starting eval on {sampler} set for best model at {model_dir}...')

    return eval.eval(-1, episodes=-1, use_best_agent=True)


def main(model_dir, *, device='cpu'):
    model_name = split_path(model_dir)[-1]
    print(f'Model name: {model_name}')
    spec = parse_model_name(model_name)
    pprint(spec)
    assert spec['progression_mode'] in {'kernel', 'dfa_naive'}
    assert '_mlp_kernel_encoder' in model_name or spec['progression_mode'] != 'kernel'

    sampler = spec['sampler']
    env = spec['env']
    progression_mode = spec['progression_mode']

    dataset_name, kernel = parse_dataset_sampler(sampler)
    print(f'Dataset_name: {dataset_name}, kernel: {kernel}')

    splits = {
        'train': f'Dataset_{dataset_name}_{kernel}_noshuffle',
        'validation': f'Dataset_{dataset_name}_{kernel}_test_noshuffle',
        'test': f'Dataset_{dataset_name}test_{kernel}_noshuffle',
    }

    n_formulas = None
    returns = {}
    steps = {}
    for split, split_sampler in splits.items():
        print(f'{split} sampler: {split_sampler}')
        returns[split], steps[split], model_frames = do_eval(env, model_name, model_dir, split_sampler, progression_mode, device)
        if n_formulas is None:
            n_formulas = get_n_formulas(model_frames, model_dir)
        else:
            assert n_formulas == get_n_formulas(model_frames, model_dir), f'Inconsistent n_formulas'

    with open(os.path.join(model_dir, 'train', 'eval.pkl'), 'wb') as f:
        pickle.dump(dict(returns=returns, steps=steps), f)

    print(f'Best model has been trained on the first {n_formulas} formulas in the curriculum.')

    with open(os.path.join(model_dir, 'train', 'eval.csv'), 'w') as f:
        f.write('split,return,n_formulas\n')
        for split, _ in splits.items():
            mean_return = np.mean(returns[split])
            print(f'({split} set) Return (mean): {mean_return:.2f}')
            f.write(f'{split},{mean_return:.2f},{n_formulas}\n')


if __name__ == '__main__':
    clize.run(main)
