"""
SentinelPay fraud scoring model: multi-head Graph Attention (GATv2-style) over the
transaction subgraph + a Transformer encoder over the resulting node embeddings, with
an optional quantum-enhanced scoring head.

Implemented as dense-adjacency attention (no torch-geometric dependency) so the service
installs cleanly for judges/reviewers spinning it up on a laptop -- the math is the same
GATv2 attention mechanism (Brody et al. 2021: score = a^T LeakyReLU(W[h_i || h_j]),
computed *after* the linear projection, which is what fixes the static-attention problem
in the original GAT). This scales fine at real-time-scoring size: each transaction is
scored against a local neighborhood of tens of accounts, not the whole ledger.

Set SENTINELPAY_QUANTUM_HEAD=1 to route the pooled embedding through a variational quantum
circuit (PennyLane) before the final classifier head, matching our published HQGT research
architecture. Falls back to a classical head automatically if pennylane isn't installed.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """Multi-head GATv2 attention over a dense adjacency mask."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.15, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.W_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))

    def forward(self, x, adj_mask):
        # x: [N, in_dim]   adj_mask: [N, N] boolean (adj_mask[i, j] = edge j -> i)
        N = x.size(0)
        h_src = self.W_src(x).view(N, self.heads, self.out_dim)  # message senders (j)
        h_dst = self.W_dst(x).view(N, self.heads, self.out_dim)  # message receivers (i)

        # e[i, j, head] = a^T LeakyReLU(h_dst_i + h_src_j)   (GATv2 formulation)
        combo = h_dst.unsqueeze(1) + h_src.unsqueeze(0)  # [N, N, heads, out_dim]
        e = self.leaky_relu(combo)
        e = torch.einsum("ijhd,hd->ijh", e, self.attn)  # [N, N, heads]

        mask = adj_mask.unsqueeze(-1)  # [N, N, 1]
        e = e.masked_fill(~mask, float("-inf"))
        # self-loops so isolated nodes still get a valid softmax
        self_mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(-1)
        e = torch.where(self_mask, torch.zeros_like(e), e)

        alpha = F.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, h_src)  # [N, heads, out_dim]
        out = out.reshape(N, self.heads * self.out_dim) if self.concat else out.mean(dim=1)
        return out + self.bias, alpha  # return attention weights for explainability


class _ClassicalHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class _QuantumHead(nn.Module):
    """Variational quantum circuit head (angle-embeds a compressed feature vector into
    n_qubits, applies entangling rotation layers, reads out Pauli-Z expectations)."""

    def __init__(self, dim, n_qubits=6, n_layers=2):
        super().__init__()
        import pennylane as qml  # local import: optional dependency

        self.pre = nn.Linear(dim, n_qubits)
        self.n_qubits, self.n_layers = n_qubits, n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="backprop")
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.post = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.tanh(self.pre(x)) * math.pi
        q_out = self.qlayer(x)
        return self.post(q_out)


class SentinelFraudNet(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, heads=4, quantum_head=None):
        super().__init__()
        self.gat1 = GATv2Layer(node_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATv2Layer(hidden_dim * heads, hidden_dim, heads=heads, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)

        use_quantum = quantum_head if quantum_head is not None else os.environ.get("SENTINELPAY_QUANTUM_HEAD") == "1"
        if use_quantum:
            try:
                self.head = _QuantumHead(hidden_dim)
                self.head_type = "quantum"
            except ImportError:
                self.head = _ClassicalHead(hidden_dim)
                self.head_type = "classical (pennylane not installed, fell back)"
        else:
            self.head = _ClassicalHead(hidden_dim)
            self.head_type = "classical"

    def forward(self, x, adj_mask, return_attention=False):
        h, alpha1 = self.gat1(x, adj_mask)
        h = F.elu(self.norm1(h))
        h, alpha2 = self.gat2(h, adj_mask)
        h = F.elu(self.norm2(h))

        h_seq = self.temporal(h.unsqueeze(0)).squeeze(0)  # relational-temporal context
        logits = self.head(h_seq).squeeze(-1)  # [N]

        if return_attention:
            return logits, {"layer1": alpha1.detach(), "layer2": alpha2.detach()}
        return logits


def edges_to_adj_mask(edge_index, n_nodes, device="cpu"):
    """edge_index: np.ndarray [2, E] with edge_index[:,k] = (src, dst). Returns [N,N] bool
    where mask[i, j] = True if there's an edge j -> i (message flows into i)."""
    mask = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
    if edge_index.size > 0:
        src, dst = edge_index[0], edge_index[1]
        mask[dst, src] = True
    return mask


def load_model(node_dim, ckpt_path=None, device="cpu"):
    model = SentinelFraudNet(node_dim=node_dim)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device)
    model.eval()
    return model
