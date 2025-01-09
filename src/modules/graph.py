import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): Dimensionality of embeddings for nodes and edges.
        out_dim (int): Dimensionality of the output embeddings.
        aggr (str): Aggregation method, default is "add".

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # Multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(2 * emb_dim, out_dim)
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # Add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self._dec_type = gnn_type 
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr="add")
        else:
            raise NotImplementedError(f"{gnn_type} is not supported. Only 'gin' is available.")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)    
        self.activation = torch.nn.PReLU() 
        self.temp = 0.2

    def forward(self, x, edge_index, edge_attr):
        x = self.activation(x)
        x = self.enc_to_dec(x)
        out = self.conv(x, edge_index, edge_attr)
        return out

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): The number of GNN layers.
        emb_dim (int): Dimensionality of embeddings.
        JK (str): Jumping Knowledge method: "last", "concat", "max", or "sum".
        drop_ratio (float): Dropout rate.
        gnn_type (str): Type of GNN, only "gin" is supported.

    Output:
        Node representations.
    """
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of GINConv layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))

        # List of batch norms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("Unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # Remove ReLU for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jumping Knowledge (JK)
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation, _ = torch.max(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)

        return node_representation

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))

class DiscreteGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): The number of GNN layers.
        emb_dim (int): Dimensionality of embeddings.
        num_tokens (int): Number of tokens for discretization.
        JK (str): Jumping Knowledge method: "last", "concat", "max", or "sum".
        temperature (float): Temperature parameter for soft assignments.
        drop_ratio (float): Dropout rate.
        gnn_type (str): Type of GNN, only "gin" is supported.

    Output:
        Node representations.
    """
    def __init__(self, num_layer, emb_dim, num_tokens, JK="last", 
                 temperature=0.9, drop_ratio=0, gnn_type="gin"):
        super(DiscreteGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.num_tokens = num_tokens
        self.temperature = temperature

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of GINConv layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer - 1):
            self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
        self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))    

        # List of batch norms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("Unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # Remove ReLU for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jumping Knowledge (JK)
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation, _ = torch.max(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)

        return node_representation

    @torch.no_grad()
    def get_codebook_indices(self, *argv):
        logits = self(*argv)
        codebook_indices = logits.argmax(dim=-1)
        return codebook_indices

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): The number of GNN layers.
        emb_dim (int): Dimensionality of embeddings.
        num_tasks (int): Number of tasks in multi-task learning scenario.
        JK (str): Jumping Knowledge method: "last", "concat", "max", or "sum".
        drop_ratio (float): Dropout rate.
        graph_pooling (str): "sum", "mean", "max", "attention", or "set2set".
        gnn_type (str): Type of GNN, only "gin" is supported.

    See:
        https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kinds of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling.startswith("set2set"):
            set2set_iter = int(graph_pooling[len("set2set"):]) if graph_pooling[len("set2set"):].isdigit() else 1
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 2 if graph_pooling.startswith("set2set") else 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("Unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        return self.pool(node_representation, batch), node_representation

