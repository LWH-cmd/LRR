"""CitationFull split, extracted without changing its scientific operations."""
import math
import torch
from torch_geometric.utils import is_undirected, negative_sampling

def split_edges_no_neg_adj_mask(
    data,
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
):
    """Exact non-KG branch of mother ``train_test_split_edges_no_neg_adj_mask``."""

    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = data.edge_weight = None
    data.edge_year = data.edge_type = None

    mask = row < col
    row, col = row[mask], col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    perm = torch.randperm(row.size(0))
    row = row[perm]
    col = col[perm]

    r, c = row[n_v + n_t :], col[n_v + n_t :]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        raise RuntimeError(
            "Mother branch references undefined 'out' when edge_attr is present"
        )
    if is_undirected(data.train_pos_edge_index):
        raise RuntimeError("directed one-way train_pos_edge_index expected")

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1],
    )

    r, c = row[n_t : n_t + n_v], col[n_t : n_t + n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1],
    )
    return data
