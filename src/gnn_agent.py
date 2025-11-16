"""
First Agent
"""
from .base_agent import BaseAgent

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy  as np# type: ignore
from torch_geometric.nn import GCNConv, SAGEConv # type: ignore
from torch_geometric.data import Data #type: ignore

class Agent(nn.Module):
    """ GNN-based multi-agent intelligent system """
    
    def __init__(
            self, 
            num_agents,
            in_dim,
            hidden_dim,
            out_dim,
        ):
        super().__init__()
        self.num_agents = num_agents

        #intialize GNN model (encoder)
        self.gnn = AgentGNN(in_dim, hidden_dim)

        #initializae Agent Policy
        self.policy = AgentPolicy(hidden_dim, out_dim)

    def forward(self, DATA):
        data = DATA.clone()
        
        embeddings = self.gnn(data.x, data.edge_index)
        # policy now returns (means, values)
        means, values = self.policy(embeddings)
        return means, values


class AgentPolicy(nn.Module):

    def __init__(
            self, 
            hidden_dim,
            out_dim
        ):
        super().__init__()
        self.hidden_dimensions = hidden_dim
        self.output_dimensions = out_dim
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.hidden_dimensions),
            nn.GELU(),
            nn.Linear(self.hidden_dimensions, self.output_dimensions)
        )
        # value head for critic (per-node scalar value)
        self.value_head = nn.Linear(self.hidden_dimensions, 1)
        # learnable log std for Gaussian policy (per action-dim)
        self.log_std = nn.Parameter(torch.zeros(self.output_dimensions) + 0.1)

    def forward(self, embeddings):
        actions = self.output_layer(embeddings)
        # value prediction per agent/node
        values = self.value_head(embeddings).squeeze(-1)
        return actions, values

class AgentGNN(nn.Module):
    def __init__(
            self,
            in_dim, 
            hidden_dim
        ):
        super().__init__()
        self.input_dimensions = in_dim
        self.hidden_dimensions = hidden_dim

        # input layer
        self.input_layer = GCNConv(self.input_dimensions, self.hidden_dimensions)

        # hidden layers
        self.hidden_layer = SAGEConv(self.hidden_dimensions, self.hidden_dimensions, aggr='mean')
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, feat, ei):
        h = self.input_layer(feat, ei)
        h = F.relu(h)
        #h = self.ln1(h)

        h = self.hidden_layer(h, ei)
        h = F.relu(h)
        #h = self.ln2(h)

        return h
    
def obs2graph(observations) -> Data:
    """
    Converts agent observations dict -> PyG Data object
    Normalizes per-agent features to zero mean and unit std
    """
    agent_obs = list(observations.values())
    agent_tensors = []

    for obs in agent_obs:
        flat = torch.tensor(obs, dtype=torch.float32).flatten()
        # normalize
        mean = flat.mean()
        std = flat.std(unbiased=False) + 1e-8
        flat = (flat - mean) / std
        agent_tensors.append(flat)

    x = torch.stack(agent_tensors, dim=0)  # [num_agents, obs_dim]

    # Fully connected graph excluding self-loops
    num_nodes = x.shape[0]
    src = torch.repeat_interleave(torch.arange(num_nodes), num_nodes)
    dst = torch.tile(torch.arange(num_nodes), (num_nodes,))
    ei = torch.stack([src, dst], dim=0)
    ei = ei[:, src != dst]

    return Data(x, ei)