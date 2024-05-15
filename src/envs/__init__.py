from gym.envs.registration import register

from envs.toymc.toymc import ToyMCEnv

__all__ = ["ToyMCEnv"]

### ToyMC Envs

register(
    id='ToyMC-v0',
    entry_point='envs.toymc.toymc:ToyMCEnv'
)

