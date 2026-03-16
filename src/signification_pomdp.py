import jax, chex
import copy
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
from belief_representations import *
from distributions import *


class ImageSigPOMDP(MultiAgentEnv):
    def __init__(self, dataset) -> None:
        super().__init__(num_agents=2)

        # Load in dataset
        self.images, self.labels = dataset

