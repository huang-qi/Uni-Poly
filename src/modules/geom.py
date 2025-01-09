import torch
import os
from torch_geometric.nn.models import SchNet
from typing import Optional, Callable
from torch import nn, Tensor

class SchNetEncoder(SchNet):
    """
    Encoder version of SchNet that outputs high-dimensional graph representations instead of scalar values.
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'mean',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[Tensor] = None,
        load_from_pretrain: Optional[str] = '../pretrained_models/schnet_qm9_model.pt',
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=interaction_graph,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
            atomref=atomref,
        )
        
        # Remove regression output layers
        # self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        # self.act = ShiftedSoftplus()
        # self.lin2 = Linear(hidden_channels // 2, 1)
        
        if load_from_pretrain is not None:
            self.load_pretrained_weights(load_from_pretrain)
            
    def load_pretrained_weights(self, pretrain_path: str):
        """
        Load weights from pretrained model path.

        Args:
            pretrain_path (str): File path to pretrained model.
        """
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrain_path}")
        
        # Load pretrained weights
        state_dict = torch.load(pretrain_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


    def forward(self, z: Tensor, pos: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass, returns graph-level high-dimensional representations.
        
        Args:
            z (torch.Tensor): Atomic numbers for each atom, shape [num_atoms].
            pos (torch.Tensor): Coordinates for each atom, shape [num_atoms, 3].
            batch (torch.Tensor, optional): Batch indices, shape [num_atoms]. Defaults to None.
        
        Returns:
            torch.Tensor: Graph-level embeddings, shape [num_graphs, hidden_channels].
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)


        # h = self.lin1(h)
        # h = self.act(h)
        # h = self.lin2(h)

        if self.dipole:
            # Calculate center of mass
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        # Use readout to generate graph-level embeddings
        graph_embedding = self.readout(h, batch, dim=0)
        # if self.dipole:
        #     out = torch.norm(out, dim=-1, keepdim=True)

        # if self.scale is not None:
        #     out = self.scale * out

        return graph_embedding

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


