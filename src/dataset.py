from contextlib import contextmanager
import json
import os
import pickle
from pprint import pprint
import random
import select
import sys
import termios
from typing import List
import clize
import numpy as np
import torch
from tqdm import tqdm
import yaml

from FiniteStateMachine import LTL2DFAError, MooreMachine
from ltl2dfa import ltl2dfa
from ltl_samplers import getLTLSampler, alphabetical_propositions
from characteristic_set import characteristic_set as get_characteristic_set
from characteristic_set import characteristic_set_no_subtraces as get_characteristic_set_no_subtraces
from utils.other import set_seed


def save_verbose(obj, path, name):
    with open(path, 'wb') as f:
        print(f'Saving {name} to {path}...', end='')
        pickle.dump(obj, f)
        print('Done.')

def check_existing(path, ask_abort=True):
    if os.path.isfile(path):
        if ask_abort and input(f'File {path} already exists. Overwrite? [y/N] ').lower() != 'y':
            print('Aborting.')
            sys.exit(0)
        return True
    return False


def formulas(dataset, config):
    formulas_path = os.path.join(dataset, 'formulas.pkl')
    automata_path = os.path.join(dataset, 'automata.pkl')
    check_existing(formulas_path)
    check_existing(automata_path)

    print(f'Generating {config["n_formulas"]} formulas using {config["sampler"]} sampler and propositions {config["propositions"]}...')
    if "disjoint_from" in config:
        print(f'Ensuring that the generated formulas are disjoint from the formulas in {config["disjoint_from"]}...')
        _, _, disjoint_formulas, _ = load_dataset(config["disjoint_from"])
    else:
        disjoint_formulas = ()

    sampler = getLTLSampler(config["sampler"], config["propositions"])
    set_seed(config['seed'])
    formulas = []
    for _ in tqdm(range(config["n_formulas"])):
        formula = None
        # Ensure that the generated formulas are unique
        while formula is None:
            formula = sampler.sample()
            if formula in formulas or formula in disjoint_formulas:
                formula = None
        formulas.append(formula)
    save_verbose(formulas, formulas_path, 'formulas')

    print('Computing the DFA of each formula...')
    automata = [ltl2dfa(formula, symbols=config["propositions"]) for formula in tqdm(formulas)]
    save_verbose(automata, automata_path, 'automata')


# traces are tuple of symbols
def startswith(trace, subtrace):
    if len(trace) < len(subtrace):
        return False
    return all(trace[i] == subtrace[i] for i in range(len(subtrace)))

def characteristic_set(dataset, config):
    characteristic_set_path = os.path.join(dataset, 'characteristic_set.pkl')
    check_existing(characteristic_set_path)

    with open(os.path.join(dataset, 'automata.pkl'), 'rb') as f:
        automata = pickle.load(f)
    automata = automata[:config['n_train_formulas']]

    if not config['characteristic_set']['remove_subtraces']:
        print(f'Computing the characteristic sets of the {len(automata)} train automata...')
        char_set = []
        for a in tqdm(automata):
            char_set += list(get_characteristic_set(a.transitions))
        print(f'Characteristic set has {len(char_set)} traces (with repetitions).')
        if config['characteristic_set']['remove_duplicates']:
            char_set = list(set(char_set))
            print(f'After removing repetitions, the characteristic set has {len(char_set)} traces.')
    else:
        print(f'Computing the characteristic set of the {len(automata)} train automata (no subtraces)...')
        char_set = set()
        pbar = tqdm(automata)
        for a in pbar:
            char_set = char_set.union(get_characteristic_set_no_subtraces(a.transitions))
            pbar.set_postfix({'n_traces': len(char_set)})
        char_set = list(char_set)
        print(f'Characteristic set has {len(char_set)} traces (no subtraces).')
    print(f'Shuffling...', end='', flush=True)
    random.shuffle(char_set)
    print('Done.')
    if "max_traces" in config["characteristic_set"]:
        print(f'Using the first {config["characteristic_set"]["max_traces"]} traces')
        char_set = char_set[:config["characteristic_set"]["max_traces"]]
    else:
        assert "max_total_symbols" in config["characteristic_set"]
        print(f'Using the first traces with a total of {config["characteristic_set"]["max_total_symbols"]} symbols...', end='', flush=True)
        new_char_set = []
        total_symbols = 0
        i = 0
        while total_symbols < config["characteristic_set"]["max_total_symbols"]:
            trace = char_set[i]
            new_char_set.append(trace)
            total_symbols += len(trace)
            i += 1
        print(f'{len(new_char_set)} traces kept.')
        char_set = new_char_set
    print('Done.')
    save_verbose(char_set, characteristic_set_path, 'characteristic set')


def try_sample_dfa(sampler, propositions, tries=100):
    # dfa = None
    # while dfa is None:
    #     try:
    #         dfa = ltl2dfa(sampler.sample(), symbols=propositions)
    #     except LTL2DFAError as e:
    #         dfa = None
    #         tries -= 1
    #         if tries == 0:
    #             raise e
    # return dfa
    return ltl2dfa(sampler.sample(), symbols=propositions)


def references(dataset, config):
    reference_automata_path = os.path.join(dataset, 'reference_automata.pkl')
    check_existing(reference_automata_path)

    set_seed(config['seed'])
    if 'n_reference_states' in config:
        print(f'Generating {config["n_references"]} reference automata with {config["n_reference_states"]} states...', end='')
        reference_automata = [
            MooreMachine(config["n_reference_states"], len(config["propositions"]), None, dictionary_symbols=config["propositions"])
            for _ in range(config["n_references"])
        ]
    elif 'sampler' in config:
        print(f'Generating {config["n_references"]} reference automata from distribution {config["sampler"]}...')
        sampler = getLTLSampler(config["sampler"], config["propositions"])
        reference_automata = [try_sample_dfa(sampler, config["propositions"]) for _ in tqdm(range(config["n_references"]))]
    else:
        raise ValueError('Either n_reference_states or sampler must be present in the config.')
    print('Done.')
    save_verbose(reference_automata, reference_automata_path, 'reference automata')


def kernel(dataset, config):
    with open(os.path.join(dataset, 'automata.pkl'), 'rb') as f:
        automata = pickle.load(f)

    for kernel, kernel_config in config["kernels"].items():
        repr_path = os.path.join(dataset, f'kernel_{kernel}.pkl')
        if check_existing(repr_path, ask_abort=False):
            kernel_reprs = load_kernel(dataset, kernel)
            if len(kernel_reprs) == len(automata):
                print(f'Kernel {kernel} already computed. Skipping...')
                continue
            print(f'Kernel {kernel} partially computed. Resuming...')
        else:
            kernel_reprs = []

        with open(os.path.join(kernel_config["references"], 'config.yaml'), 'rb') as f:
            references_config = yaml.full_load(f)
        if references_config["propositions"] != config["propositions"]:
            raise ValueError('The propositions of the reference automata must match the propositions of the dataset.')
        # TODO what other configs must be checked?

        with open(os.path.join(kernel_config["references"], 'reference_automata.pkl'), 'rb') as f:
            reference_automata = pickle.load(f)
        print(f'Computing the kernel representation {kernel} of each DFA from the references at {kernel_config["references"]}...')

        traces = kernel_config.get("traces")
        if traces is not None and os.path.isfile(traces):
            print(f'Using traces from {traces}.')
            with open(traces, 'rb') as f:
                traces = pickle.load(f)
        elif traces == 'characteristic':
            print(f'Using the characteristic set of traces for the train formulas.')
            with open(os.path.join(dataset, 'characteristic_set.pkl'), 'rb') as f:
                traces = pickle.load(f)
        elif traces == 'all':
            traces = None
        elif traces is not None:
            raise ValueError('traces must be either a file path, "characteristic", "all" or None.')

        if traces is not None:
            print(f'Using {len(traces)} traces.')
        else:
            print(f'Using all traces of max length {kernel_config["max_trace_length"]}.')

        i_start = len(kernel_reprs)
        print(f'Starting from {i_start} automata.')
        for a in until_esc(tqdm(automata[i_start:])):
            kernel_reprs.append(a.calculate_state_representation_multiple_references(
                reference_automata,
                max_trace_len=kernel_config.get("max_trace_length"),
                all_traces=traces,
            ))
        save_verbose(kernel_reprs, repr_path, 'kernel representations')


def load_dataset(dataset, kernel=None):
    with open(os.path.join(dataset, 'config.yaml'), 'r') as f:
        config = yaml.full_load(f)
    with open(os.path.join(dataset, 'automata.pkl'), 'rb') as f:
        automata = pickle.load(f)
    with open(os.path.join(dataset, 'formulas.pkl'), 'rb') as f:
        formulas = pickle.load(f)
    if os.path.isfile(os.path.join(dataset, 'characteristic_set.pkl')):
        with open(os.path.join(dataset, 'characteristic_set.pkl'), 'rb') as f:
            char_set = pickle.load(f)
    else:
        char_set = None
    ret = (config, automata, formulas, char_set)
    if kernel is not None:
        with open(os.path.join(dataset, f'kernel_{kernel}.pkl'), 'rb') as f:
            kernel_reprs = pickle.load(f)
        ret += (kernel_reprs,)
    return ret

def load_kernel(dataset, kernel):
    with open(os.path.join(dataset, f'kernel_{kernel}.pkl'), 'rb') as f:
        reprs = pickle.load(f)
    return reprs

def getattr_here(name):
    return getattr(sys.modules[__name__], name)

def main(command, dataset):
    with open(os.path.join(dataset, 'config.yaml'), 'r') as f:
        config = yaml.full_load(f)
    pprint(config)

    os.makedirs(dataset, exist_ok=True)

    getattr_here(command)(dataset, config)


# utils

@contextmanager
def unbuffered_stdin():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        settings = old_settings.copy()
        settings[3] = settings[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def until_esc(it):
    with unbuffered_stdin():
        for elem in it:
            rready, _, _ = select.select([sys.stdin], [], [], 0)
            if len(rready) > 0:
                char = os.read(sys.stdin.fileno(), 1).decode("utf-8")
                if ord(char) == 27: # ASCII ESC
                    break
            yield elem



if __name__ == '__main__':
    clize.run(main)