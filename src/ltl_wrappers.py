import numpy as np
import gym
from gym import spaces
from ltl_samplers import getLTLSampler, SequenceSampler


class KernelDfaEnv(gym.Wrapper):

    def __init__(self, env, ltl_sampler=None):
        super().__init__(env)
        self.symbols = self.env.get_symbols()
        self.sampler = getLTLSampler(ltl_sampler, self.symbols)
        # self.formula_history = []
        # TODO check, shouldn't this also have the 'kernel' field?
        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.last_return = 0


    def get_kernel(self):
        return self.kernel_state_repr[self.automaton_state]


    def get_reward(self):
        return self.automaton.rewards[self.automaton_state]

    def get_done(self):
        return self.automaton.acceptance[self.automaton_state]


    def step_automaton(self):
        self.automaton_state = self.automaton.transitions[self.automaton_state][self.env.get_symbol()]


    def get_last_automaton_and_kernel_repr(self):
        if hasattr(self.sampler, 'get_last_automaton') and hasattr(self.sampler, 'get_last_kernel_representation'):
            automaton = self.sampler.get_last_automaton()
            repr = self.sampler.get_last_kernel_representation()
            return automaton, repr
        else:
            raise ValueError(f'sampler {self.sampler} does not have get_last_automaton and get_last_kernel_representation methods')


    def reset(self):
        self.obs = self.env.reset()

        # Defining an LTL goal
        if hasattr(self.sampler, 'update_curriculum') and self.sampler.curriculum:
            self.sampler.update_curriculum(self.last_return)
        self.ltl_goal = self.sample_ltl_goal()
        # self.formula_history.append(self.ltl_goal)
        self.automaton, self.kernel_state_repr = self.get_last_automaton_and_kernel_repr()
        # print('done parsing ltl')
        self.automaton_state = 0 # NOTE assume 0 is always the initial state

        # Adding the ltl goal to the observation
        ltl_obs = {'features': self.obs, 'kernel': self.get_kernel()}

        self.last_return = 0
        return ltl_obs


    def step(self, action):
        int_reward = 0 # reward for intermediate states
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        s_a = self.automaton_state
        sym = self.env.get_symbol()
        self.step_automaton()
        if s_a is not None and self.automaton_state is None:
            print(f'automaton state {s_a} has no transition for symbol {sym}\ntrans: {self.automaton.transitions}, acc: {self.automaton.acceptance}, trans[s_a]: {self.automaton.transitions[s_a]}, trans[s_a][sym]: {self.automaton.transitions[s_a][sym]}')
            raise Exception(f'automaton state {s_a} has no transition for symbol {self.env.get_symbol()}')
        self.obs = next_obs

        ltl_reward = self.get_reward()

        # Computing the new observation and returning the outcome of this action
        ltl_obs = {'features': self.obs,'kernel': self.get_kernel()}

        reward  = original_reward + ltl_reward
        done    = env_done or self.get_done()
        self.last_return += reward
        return ltl_obs, reward, done, info


    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            # TODO is this timeout relevant to us?
            self.env.timeout = 25 # 10 * length

        return formula


class NaiveDfaEnv(gym.Wrapper):
    def __init__(self, env, ltl_sampler=None):
        """
        DFA-Naive environment
        --------------------
        It adds an LTL objective to the current environment, encoded as
        a DFA. The observations become a dictionary with an added
        "automaton_state" field specifying the current state of the DFA,
        i.e. an integer from 0 to num_states - 1.

        It needs the env to implement the following functions (in
        addition to the usual gym interface):
        - get_symbol: returns the current symbol (according to the env
          state). Must return an int, to be used as index for automaton
          transitions
        - get_symbols: returns the list of symbols (strings) defined by
          the env
        """
        super().__init__(env)
        self.symbols = self.env.get_symbols()
        self.sampler = getLTLSampler(ltl_sampler, self.symbols)
        # TODO check, shouldn't this also have the 'kernel' field?
        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.last_return = 0


    def get_reward(self):
        return self.automaton.rewards[self.automaton_state]

    def get_done(self):
        return self.automaton.acceptance[self.automaton_state]


    def step_automaton(self):
        self.automaton_state = self.automaton.transitions[self.automaton_state][self.env.get_symbol()]


    def get_last_automaton_and_kernel_repr(self):
        if hasattr(self.sampler, 'get_last_automaton') and hasattr(self.sampler, 'get_last_kernel_representation'):
            automaton = self.sampler.get_last_automaton()
            repr = self.sampler.get_last_kernel_representation()
            return automaton, repr
        else:
            raise ValueError(f'sampler {self.sampler} does not have get_last_automaton and get_last_kernel_representation methods')


    def reset(self):
        self.obs = self.env.reset()

        # Defining an LTL goal
        if hasattr(self.sampler, 'update_curriculum') and self.sampler.curriculum:
            self.sampler.update_curriculum(self.last_return)
        self.ltl_goal = self.sample_ltl_goal()
        self.automaton, _ = self.get_last_automaton_and_kernel_repr()
        # print('done parsing ltl')
        self.automaton_state = 0 # NOTE assume 0 is always the initial state

        # Adding the ltl goal to the observation
        ltl_obs = {'features': self.obs, 'automaton_state': (self.automaton_state,)}

        self.last_return = 0
        return ltl_obs


    def step(self, action):
        int_reward = 0 # reward for intermediate states
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        s_a = self.automaton_state
        sym = self.env.get_symbol()
        self.step_automaton()
        if s_a is not None and self.automaton_state is None:
            print(f'automaton state {s_a} has no transition for symbol {sym}\ntrans: {self.automaton.transitions}, acc: {self.automaton.acceptance}, trans[s_a]: {self.automaton.transitions[s_a]}, trans[s_a][sym]: {self.automaton.transitions[s_a][sym]}')
            raise Exception(f'automaton state {s_a} has no transition for symbol {self.env.get_symbol()}')
        self.obs = next_obs

        ltl_reward = self.get_reward()

        # Computing the new observation and returning the outcome of this action
        ltl_obs = {'features': self.obs, 'automaton_state': (self.automaton_state,)}

        reward  = original_reward + ltl_reward
        done    = env_done or self.get_done()
        self.last_return += reward
        return ltl_obs, reward, done, info


    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            # TODO is this timeout relevant to us?
            self.env.timeout = 25 # 10 * length

        return formula