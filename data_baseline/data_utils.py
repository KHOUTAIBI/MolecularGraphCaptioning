"""
Data loading and processing utilities for molecule-text retrieval.
Includes dataset classes and data loading functions.
"""
from typing import Dict
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Feature maps for atom and bond attributes
# =========================================================
from typing import Dict, List, Any

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# =========================================================
# Load precomputed text embeddings
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        
    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    id2desc = {}
    for graph in graphs:
        id2desc[graph.id] = graph.description
    
    return id2desc


# =========================================================
# Dataset that loads preprocessed graphs and text embeddings
# =========================================================
class PreprocessedGraphDataset(Dataset):
    """
    Dataset that loads pre-saved molecule graphs with optional text embeddings.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        emb_dict: Dictionary mapping ID to text embedding tensors (optional)
    """
    def __init__(self, graph_path: str, emb_dict: Dict[str, torch.Tensor] = None):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.emb_dict = emb_dict
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.emb_dict is not None:
            id_ = graph.id
            text_emb = self.emb_dict[id_]
            return graph, text_emb
        else:
            return graph


def collate_fn(batch):
    """
    Collate function for DataLoader to batch graphs with optional text embeddings.
    
    Args:
        batch: List of graph Data objects or (graph, text_embedding) tuples
        
    Returns:
        Batched graph or (batched_graph, stacked_text_embeddings)
    """
    if isinstance(batch[0], tuple):
        graphs, text_embs = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        text_embs = torch.stack(text_embs, dim=0)
        return batch_graph, text_embs
    else:
        return Batch.from_data_list(batch)

