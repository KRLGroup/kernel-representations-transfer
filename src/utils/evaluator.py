import time
import torch
from tqdm import tqdm
from ltl_samplers import DatasetSampler
from torch_ac.utils.penv import ParallelEnv
#import tensorboardX

import utils
import argparse
import datetime


class Eval:
    def __init__(self, env, model_name, ltl_sampler,
                seed=0, device="cpu", argmax=True,
                num_procs=1, ignoreLTL=False, progression_mode=True,
                gnn=None, recurrence=1, dumb_ac = False, discount=0.99,
                legacy_kernel_encoder=False, render=False):

        self.env = env
        self.ltl_sampler = ltl_sampler
        self.device = device
        self.argmax = argmax
        self.num_procs = num_procs
        self.ignoreLTL = ignoreLTL
        self.progression_mode = progression_mode
        self.gnn = gnn
        self.recurrence = recurrence
        self.dumb_ac = dumb_ac
        self.discount = discount
        self.legacy_kernel_encoder = legacy_kernel_encoder
        self.render = render
        if self.render:
            assert num_procs == 1, "Rendering only supported for single process"

        self.model_dir = utils.get_model_dir(model_name)

        # Load environments for evaluation
        eval_envs = []
        for i in range(self.num_procs):
            eval_envs.append(utils.make_env(env, progression_mode, ltl_sampler, seed))

        eval_envs[0].reset()

        self.eval_envs = ParallelEnv(eval_envs)




    def eval(self, num_frames, episodes=100, stdout=True, use_best_agent=False):

        if episodes == -1:
            assert len(self.eval_envs.envs) == 1, "Only one environment supported when evaluating the whole dataset"
            sampler = self.eval_envs.envs[0].sampler
            assert isinstance(sampler, DatasetSampler), "When evaluating the whole dataset, the sampler must be a DatasetSampler"
            episodes = len(sampler.items)
            sampler.reset()
            print(f'Using the whole dataset for evaluation ({episodes} episodes)')

        # Load agent
        
        # print(f'Loading agent from {self.model_dir}')
        agent = utils.Agent(self.eval_envs.envs[0], self.eval_envs.observation_space,
            self.eval_envs.action_space, self.model_dir + "/train", 
            self.progression_mode, 
            device=self.device, argmax=self.argmax,
            num_envs=self.num_procs,
            legacy_kernel_encoder=self.legacy_kernel_encoder,
            load_best_status=use_best_agent,
        )


        # Run agent
        start_time = time.time()

        obss = self.eval_envs.reset()
        log_counter = 0

        log_episode_return = torch.zeros(self.num_procs, device=self.device)
        log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        # Initialize logs
        logs = {"num_frames_per_episode": [], "return_per_episode": []}
        with tqdm(total=episodes) as pbar:
            while log_counter < episodes:
                actions = agent.get_actions(obss)
                if self.render:
                    self.eval_envs.envs[0].render()
                obss, rewards, dones, _ = self.eval_envs.step(actions)

                log_episode_return += torch.tensor(rewards, device=self.device, dtype=torch.float)
                log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

                for i, done in enumerate(dones):
                    if done:
                        log_counter += 1
                        logs["return_per_episode"].append(log_episode_return[i].item())
                        logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
                        pbar.update(1)

                mask = 1 - torch.tensor(dones, device=self.device, dtype=torch.float)
                log_episode_return *= mask
                log_episode_num_frames *= mask

        end_time = time.time()


        results = (logs["return_per_episode"], logs["num_frames_per_episode"])
        if use_best_agent:
            results += (agent.loaded_status["num_frames"],)
        return results
