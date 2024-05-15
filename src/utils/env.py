import gym
import ltl_wrappers


def make_env(env_key, progression_mode, ltl_sampler, seed=None):
    env = gym.make(env_key)
    env.seed(seed)

    # Adding LTL wrappers
    if progression_mode == "kernel":
        return ltl_wrappers.KernelDfaEnv(env, ltl_sampler)
    elif progression_mode == "dfa_naive":
        return ltl_wrappers.NaiveDfaEnv(env, ltl_sampler)
