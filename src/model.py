import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch_ac

from gym.spaces import Box, Discrete

from env_model import getEnvModel
from policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, env, obs_space, action_space, legacy_kernel_encoder=False):
        super().__init__()

        self.recurrent = False

        # Decide which components are enabled
        self.use_kernel = "kernel" in obs_space
        self.use_automaton_state = "automaton_state" in obs_space
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space

        self.env_model = getEnvModel(env, obs_space)

        # Define text embedding
        if self.use_kernel:
            self.kernel_embedding_size = obs_space['kernel']
            if legacy_kernel_encoder:
                self.kernel_encoder = lambda x: x
                print(f'Using legacy kernel encoder (identity function, 0 parameters)')
            else:
                self.kernel_encoder = nn.Sequential(
                    nn.Linear(obs_space["kernel"], 64),
                    nn.Tanh(),
                    nn.Linear(64, self.kernel_embedding_size),
                    nn.Tanh()
                ).to(self.device)
                print("Kernel encoder Number of parameters:", sum(p.numel() for p in self.kernel_encoder.parameters() if p.requires_grad))

        elif self.use_automaton_state:
            self.automaton_state_embedding_size = 32
            self.automaton_state_encoder = nn.Sequential(
                nn.Linear(obs_space["automaton_state"], 64),
                nn.Tanh(),
                nn.Linear(64, self.automaton_state_embedding_size),
                nn.Tanh()
            ).to(self.device)
            print("DFA naive state encoder Number of parameters:", sum(p.numel() for p in self.automaton_state_encoder.parameters() if p.requires_grad))

       # Resize image embedding
        self.embedding_size = self.env_model.size()
        print("embedding size:", self.embedding_size)
        if self.use_kernel:
            self.embedding_size += self.kernel_embedding_size
        elif self.use_automaton_state:
            self.embedding_size += self.automaton_state_embedding_size

        # Define actor's model
        self.actor = PolicyNetwork(self.embedding_size, self.action_space, hiddens=[64, 64, 64], activation=nn.ReLU())

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)

        if self.use_kernel:
            # print(f'[0] obs.kernel.dtype: {obs.kernel.dtype}')
            embed_ltl = self.kernel_encoder(obs.kernel.float())
            # print(f'embedding shape: {embedding.shape}, embedding: {embedding}, embed_ltl shape: {embed_ltl.shape}, embed_ltl: {embed_ltl}')
            embedding = torch.cat((embedding.float(), embed_ltl), dim=1) if embedding is not None else embed_ltl

        elif self.use_automaton_state:
            embed_ltl = self.automaton_state_encoder(obs.automaton_state.float())
            # print(f'embed_ltl shape: {embed_ltl.shape}, embedding shape: {embedding.shape}, embed_ltl: {embed_ltl}, embedding: {embedding}')
            embedding = torch.cat((embedding.float(), embed_ltl), dim=1) if embedding is not None else embed_ltl

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value
