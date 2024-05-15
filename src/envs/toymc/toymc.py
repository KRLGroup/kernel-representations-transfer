import gym
from gym import spaces
import pygame
import random
import numpy as np
from FiniteStateMachine import MooreMachine


class ToyMCEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # interface for ltlwrappers
    def get_propositions(self):
        return self.get_symbols()
    def get_events(self):
        return self.dictionary_symbols[self.get_symbol()]


    def __init__(self, render_mode=None, state_type = "symbolic", train=True, size=4, ref_dfas = []):
        self.dictionary_symbols = ['a', 'b', 'c', 'd', 'e']
        self._PICKAXE = "envs/toymc/imgs/pickaxe.png"
        self._GEM = "envs/toymc/imgs/gem.png"
        self._DOOR = "envs/toymc/imgs/door.png"
        self._ROBOT = "envs/toymc/imgs/robot.png"
        self._LAVA = "envs/toymc/imgs/lava.jpg"
        self._train = train
        self.max_num_steps = 30
        self.curr_step = 0

        self.size = size  # 4x4 world
        self.window_size = 512  # size of the window

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.observation_space = spaces.MultiDiscrete([size, size])
        self.action_space = spaces.Discrete(4)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT

        self._action_to_direction = {
            0: np.array([0, 1]),  # DOWN
            1: np.array([1, 0]),  # RIGHT
            2: np.array([0, -1]),  # UP
            3: np.array([-1, 0]),  # LEFT
        }

    def get_symbols(self):
        return self.dictionary_symbols.copy()
    
    def get_symbol(self):
        return self._current_symbol()

    def reset(self, seed=None):
        self.curr_step = 0
        self._agent_location = np.array([0, 0])
        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([1, 1])
        self._exit_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = True

        if self.render_mode == "human":
            self._render_frame()

        # print('reset')

        return self._observation()

    def _current_symbol(self):
        if (self._agent_location == self._exit_location).all():
            return 2
        if (self._agent_location == self._pickaxe_location).all():
            return 0
        if (self._agent_location == self._gem_location).all():
            return 3
        if (self._agent_location == self._lava_location).all():
            return 1
        return 4

    def _observation(self):
        observation = np.array(self._agent_location)
        return observation

    def step(self, action):

        self.curr_step += 1
        done = False

        # MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        if self.render_mode == "human":
            self._render_frame()

        # print('step')

        truncated = (self.curr_step >= self.max_num_steps)
        done = truncated
        return self._observation(), 0.0, done, {}

    def render(self, mode):
        if mode in {"human", "rgb_array"}:
            self.render_mode = mode
            return self._render_frame()

    def _get_obs(self):
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        img = img[:, :, ::-1]
        obs = img
        return obs

    def _get_info(self):
        info = {
            "robot location": self._agent_location,
            "inventory": "empty"
        }
        if self._has_gem:
            info["inventory"] = "gem"
        elif self._has_pickaxe:
            info["inventory"] = "pickaxe"
        else:
            info["inventory"] = "empty"
        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            pickaxe = pygame.image.load(self._PICKAXE)
            gem = pygame.image.load(self._GEM)
            door = pygame.image.load(self._DOOR)
            robot = pygame.image.load(self._ROBOT)
            lava = pygame.image.load(self._LAVA)
            self.window.blit(canvas, canvas.get_rect())

            if self._robot_display:
                self.window.blit(robot,
                                 (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))
            if self._pickaxe_display:
                self.window.blit(pickaxe, (
                pix_square_size * self._pickaxe_location[0], pix_square_size * self._pickaxe_location[1]))
            if self._gem_display:
                self.window.blit(gem, (
                pix_square_size * self._gem_location[0], 32 + pix_square_size * self._gem_location[1]))
            self.window.blit(door, (pix_square_size * self._exit_location[0], pix_square_size * self._exit_location[1]))
            self.window.blit(lava, (
            pix_square_size * self._lava_location[0] + 2, pix_square_size * self._lava_location[1] + 2))

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()