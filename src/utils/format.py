"""
These functions preprocess the observations.
When trying more sophisticated encoding for LTL, we might have to modify this code.
"""

import os
import json
import re
import torch
import torch_ac
import gym
import numpy as np
import utils

from envs import *
from ltl_wrappers import KernelDfaEnv, NaiveDfaEnv

def get_obss_preprocessor(env, progression_mode):
    obs_space = env.observation_space

    if isinstance(env, (KernelDfaEnv, NaiveDfaEnv)): #LTLEnv Wrapped env
        wrapper = env
        env = env.unwrapped
        if isinstance(env, ToyMCEnv):
            if progression_mode == "kernel":
                n_ref_dfas = wrapper.kernel_state_repr.shape[1]
                obs_space = {"features": obs_space.shape, "kernel": n_ref_dfas}
                print(f'[get_obss_preprocessor] Num of ref dfas: {n_ref_dfas}')
                def preprocess_obss(obss, device=None):
                    # simply turn into tensors and move them to the device
                    return torch_ac.DictList({
                        'features': torch.tensor(np.array([obs["features"] for obs in obss]), device=device),
                        'kernel': torch.tensor(np.array([obs["kernel"] for obs in obss]), device=device)
                    })
            elif progression_mode == "dfa_naive":
                obs_space = {"features": obs_space.shape, "automaton_state": 1}
                def preprocess_obss(obss, device=None):
                    # simply turn into tensors and move them to the device
                    return torch_ac.DictList({
                        'features': torch.tensor(np.array([obs["features"] for obs in obss]), device=device),
                        'automaton_state': torch.tensor(np.array([obs["automaton_state"] for obs in obss]), device=device)
                    })
            else:
                assert False, "Unknown progression mode for ToyMCEnv"
        
        else:
            raise ValueError("Unknown observation space: " + str(obs_space))
    # Check if obs_space is an image space
    elif isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)
