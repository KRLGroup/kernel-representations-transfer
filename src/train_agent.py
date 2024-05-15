import argparse
from itertools import product
import os
import random
import time
import datetime
import numpy as np
import torch
from ltl_samplers import DatasetSampler
import torch_ac
import tensorboardX
import sys
import glob
from math import floor

import utils
from model import ACModel
from utils.other import set_seed

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--ltl-sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{SAMPLER}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of updates between two logs (default: 10)")
parser.add_argument("--save-interval", type=int, default=100,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, required=True,
                    help="number of frames of training (-1 means until end of curriculum)")
parser.add_argument("--checkpoint-dir", default=None)

## Evaluation parameters
parser.add_argument("--eval", action="store_true", default=False,
                    help="evaluate the saved model (default: False)")
parser.add_argument("--eval-episodes", type=int,  default=5,
                    help="number of episodes to evaluate on (default: 5)")
parser.add_argument("--eval-env", default=None,
                    help="name of the environment to train on (default: use the same \"env\" as training)")
# first one is used for determining best model
parser.add_argument("--ltl-samplers-eval", default=None, nargs='+',
                    help="the ltl formula templates to sample from for evaluation (default: use the same \"ltl-sampler\" as training)")
parser.add_argument("--eval-procs", type=int, default=1,
                    help="number of processes (default: use the same \"procs\" as training)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--progression-mode", default="dfa_naive",
                    help="either kernel (use kernel representation) or dfa_naive (use vanilla reward machines)")

args = parser.parse_args()

assert args.ltl_sampler.startswith("Dataset"), "Only DatasetSampler is supported for training."

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

default_model_name = f"{args.ltl_sampler}_{args.env}_seed:{args.seed}_epochs:{args.epochs}_bs:{args.batch_size}_fpp:{args.frames_per_proc}_procs:{args.procs}_dsc:{args.discount}_lr:{args.lr}_ent:{args.entropy_coef}_clip:{args.clip_eps}_prog:{args.progression_mode}"
if args.progression_mode == "kernel":
    default_model_name += f"_mlp_kernel_encoder:True"

model_name = args.model or default_model_name
storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir
model_dir = utils.get_model_dir(model_name, storage_dir)

pretrained_model_dir = None

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir + "/train")
csv_file, csv_logger = utils.get_csv_logger(model_dir + "/train")
tb_writer = tensorboardX.SummaryWriter(model_dir + "/train")
utils.save_config(model_dir + "/train", args)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

set_seed(args.seed)

# Set device

device = torch.device("cpu")
txt_logger.info(f"Device: {device} (cuda version: {torch.version.cuda})\n")

# Load environments

envs = []
progression_mode = args.progression_mode
for i in range(args.procs):
    envs.append(utils.make_env(args.env, progression_mode, args.ltl_sampler, args.seed))

# Sync environments
envs[0].reset()

txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir + "/train")
except OSError:
    txt_logger.info('No saved status found. Starting from scratch.\n')
    status = {"num_frames": 0, "update": 0}
else:
    txt_logger.info(f'Loaded status from previous run with keys: {list(status.keys())}.\n')

if pretrained_model_dir is not None:
    try:
        pretrained_status = utils.get_status(pretrained_model_dir)
    except:
        txt_logger.info("Failed to load pretrained model.\n")
        exit(1)

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0], progression_mode)
if "vocab" in status and preprocess_obss.vocab is not None:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded.\n")

# Load model
acmodel = ACModel(envs[0].env, obs_space, envs[0].action_space)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
    txt_logger.info("Loading model from existing run.\n")

print(f'Moving model to device {device}')
acmodel.to(device)
txt_logger.info("Model loaded.\n")
txt_logger.info("{}\n".format(acmodel))

# Load algo
algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Loading optimizer from existing run.\n")
txt_logger.info("Optimizer loaded.\n")

# init the evaluator
if args.eval:
    if args.ltl_samplers_eval is not None:
        eval_samplers = args.ltl_samplers_eval 
    else:
        eval_ltl_sampler = args.ltl_sampler
        if eval_ltl_sampler.startswith("Dataset"):
            if eval_ltl_sampler.endswith("_curriculum"):
                eval_ltl_sampler = eval_ltl_sampler.split("_curriculum")[0]
            assert not eval_ltl_sampler.endswith("_test") and not eval_ltl_sampler.endswith("_noshuffle")
            eval_ltl_sampler += '_test_noshuffle'
        eval_samplers = [eval_ltl_sampler]
    print(f"Using {eval_samplers} for evaluation.")
    eval_env = args.eval_env if args.eval_env else args.env
    eval_procs = args.eval_procs if args.eval_procs else args.procs

    evals = []
    for eval_sampler in eval_samplers:
        evals.append(utils.Eval(eval_env, model_name, eval_sampler,
                    seed=args.seed, device=device, num_procs=eval_procs,
                    progression_mode=progression_mode))


# Train model

num_frames = status["num_frames"]
update = status["update"]
best_eval = -float('inf')
all_failed = {}
curriculum_completed = False
start_time = time.time()

while num_frames < args.frames or (args.frames == -1 and not curriculum_completed):
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # check if curriculum is completed

    train_ltl_sampler = algo.env.envs[0].sampler if hasattr(algo.env.envs[0], "sampler") else None
    if isinstance(train_ltl_sampler, DatasetSampler):
        assert len(algo.env.envs) == 1, "Curriculum learning is only supported for single process."
        if train_ltl_sampler.curriculum:
            if train_ltl_sampler.cycle == 1: # dataset was exhausted once
                curriculum_completed = True
                txt_logger.info(f"Curriculum completed after {num_frames} frames.")

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)

        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], args.discount)
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["average_reward_per_step", "average_discounted_return"]
        data += [average_reward_per_step, average_discounted_return]
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ARPS: {:.3f} | ADR: {:.3f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

        if hasattr(algo.env.envs[0], "sampler"):
            sampler = algo.env.envs[0].sampler
        else:
            sampler = None
        if isinstance(sampler, DatasetSampler) and sampler.curriculum:
            assert len(algo.env.envs) == 1, "Curriculum learning is only supported for single process."
            tb_writer.add_scalar("curriculum/current_formula_idx", sampler.i, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": algo.acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab") and preprocess_obss.vocab is not None:
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir + "/train")
        txt_logger.info(f"Status saved to {model_dir}/train\n")

        if args.eval:
            tb_return_dict = {}
            tb_num_frames_dict = {}
            for i, evalu in enumerate(evals):
                eval_returns, eval_ep_lens = evalu.eval(num_frames, episodes=args.eval_episodes)                
                assert all(r in {0., 100.} for r in eval_returns), f"Eval returns were {eval_returns}"
                # record formulas failed by the model on the first evaluator
                if i == 0 and evalu.ltl_sampler.startswith("Dataset"):
                    failed = (np.isclose(np.array(eval_returns), 0)).nonzero()[0]
                    for i_formula in failed:
                        if i_formula not in all_failed:
                            all_failed[i_formula] = 0
                        all_failed[i_formula] += 1
                    utils.save_all_failed(all_failed, model_dir + "/train")
                # decide best model based on the first evaluator
                if i == 0 and np.mean(eval_returns) > best_eval:
                    utils.save_best_status(status, model_dir + "/train")
                    # utils.storage.save_pkl(formulas, model_dir + "/train/best_status_formulas.pkl")
                    check = utils.storage.file_md5(utils.get_best_status_path(model_dir + "/train"), mode='rb')
                    txt_logger.info(f"New best model found at {num_frames} frames with avg return {np.mean(eval_returns)}. Used sampler: {evalu.ltl_sampler}. Checksum: {check}")
                    best_eval = np.mean(eval_returns)
                print(f'Eval return for {evalu.ltl_sampler}: {np.mean(eval_returns)}, shape was {np.shape(eval_returns)}')
                tb_return_dict[evalu.ltl_sampler] = np.mean(eval_returns)
                tb_num_frames_dict[evalu.ltl_sampler] = np.mean(eval_ep_lens)
            if len(evals) == 1:
                tb_writer.add_scalar(f"eval_return", tb_return_dict[evals[0].ltl_sampler], num_frames)
                tb_writer.add_scalar(f"eval_num_frames", tb_num_frames_dict[evals[0].ltl_sampler], num_frames)
            else:
                tb_writer.add_scalars(f"eval_return", tb_return_dict, num_frames)
                tb_writer.add_scalars(f"eval_num_frames", tb_num_frames_dict, num_frames)
