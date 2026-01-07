#!/usr/bin/env python3
"""
T5-only Retrieval→Edit pipeline for Molecular Graph Captioning.

- Loads train/val/test molecule graphs from .pkl (PyG Data objects).
- Loads a pre-trained molecule encoder checkpoint (MolGINE) and encodes train/val/test graphs into embeddings.
- Retrieves K nearest train molecules for each train/val/test molecule (mol→mol).
- Builds editor prompts from retrieved neighbor captions (+ optional simple molecule features).
- Fine-tunes a T5 family model (t5-small / t5-base / t5-large / flan-t5-*) to rewrite a correct caption.
- Prints BLEU precision/recall/F1 on validation at each epoch.
- Generates submission CSV for test.

Requirements:
- data/{train,validation,test}_graphs.pkl
- data_utils.py from your starter code (must expose: PreprocessedGraphDataset, collate_fn, load_descriptions_from_graphs, x_map, e_map)
- A MolGINE checkpoint trained to produce embeddings (default: mol_gine_bert_clip.pt)

Run:
  python run_t5_editor.py --t5_model t5-base --mol_ckpt mol_gine_bert_clip.pt

Tips:
- If you plateau: try --t5_model t5-base or google/flan-t5-base, adjust --kedit and decoding params.
"""

import os
import argparse
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

# ---- project imports (starter) ----
from data_utils import (
    PreprocessedGraphDataset, collate_fn,
    load_descriptions_from_graphs, x_map, e_map
)

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool


# =========================
# Mol encoder (MolGINE)
# =========================
class CatFeatureEncoder(nn.Module):
    def __init__(self, sizes: List[int], emb_dim: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, emb_dim) for s in sizes])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        out = 0
        for j, emb in enumerate(self.embs):
            out = out + emb(x_cat[:, j])
        return out


class MolGINE(nn.Module):
    def __init__(self, node_dim: int = 256, out_dim: int = 768, layers: int = 4):
        super().__init__()
        node_sizes = [len(x_map[k]) for k in x_map.keys()]
        edge_sizes = [len(e_map[k]) for k in e_map.keys()]

        self.node_enc = CatFeatureEncoder(node_sizes, node_dim)
        self.edge_enc = CatFeatureEncoder(edge_sizes, node_dim)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(node_dim, 2 * node_dim),
                nn.ReLU(),
                nn.Linear(2 * node_dim, node_dim),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=node_dim))

        self.proj = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x = self.node_enc(batch.x.long())
        e = self.edge_enc(batch.edge_attr.long())
        for conv in self.convs:
            x = conv(x, batch.edge_index, e)
            x = F.relu(x)
        g = global_add_pool(x, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


@torch.no_grad()
def encode_mols(mol: nn.Module, graphs_pkl: str, device: str, batch_size: int = 256) -> Tuple[List[str], torch.Tensor]:
    ds = PreprocessedGraphDataset(graphs_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_g = []
    for graphs in tqdm(dl, desc=f"encode mol {os.path.basename(graphs_pkl)}"):
        graphs = graphs.to(device)
        all_g.append(mol(graphs).cpu())
    emb = torch.cat(all_g, 0)
    emb = F.normalize(emb, dim=-1)
    return ds.ids, emb


# =========================
# Prompting utilities
# =========================
def build_prompt(neigh_descs: List[str], feat: Optional[str] = None) -> str:
    lines = ["NEIGHBORS:"]
    for i, d in enumerate(neigh_descs, 1):
        lines.append(f"{i}) {d}")
    if feat is not None:
        lines.append(f"FEATS: {feat}")
    lines.append("TASK: write the correct caption.")
    return "\n".join(lines)


def mol_feat(graph) -> str:
    n_atoms = int(graph.x.size(0))
    n_bonds = int(graph.edge_index.size(1) // 2)
    return f"atoms={n_atoms};bonds={n_bonds}"


# =========================
# T5 dataset + training
# =========================
class T5DS(Dataset):
    def __init__(self, inputs: List[str], targets: List[str]):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int):
        return self.inputs[i], self.targets[i]


def make_collate_t5(tok, max_in: int, max_out: int):
    def collate(batch):
        x, y = zip(*batch)
        enc = tok(list(x), padding=True, truncation=True, max_length=max_in, return_tensors="pt")
        # transformers compatibility: as_target_tokenizer is deprecated in newer versions, but still works in many envs
        with tok.as_target_tokenizer():
            dec = tok(list(y), padding=True, truncation=True, max_length=max_out, return_tensors="pt")
        labels = dec["input_ids"]
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def generate_all(model, tok, inputs: List[str], device: str,
                 batch_size: int, max_in: int, max_new_tokens: int,
                 num_beams: int, length_penalty: float, no_repeat_ngram_size: int) -> List[str]:
    model.eval()
    preds: List[str] = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="generate"):
        b = inputs[i:i + batch_size]
        enc = tok(b, padding=True, truncation=True, max_length=max_in, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        preds += tok.batch_decode(out, skip_special_tokens=True)
    return preds


def bleu_f1(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    bp = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    br = sacrebleu.corpus_bleu(refs, [preds]).score / 100.0
    bf1 = 2 * bp * br / (bp + br + 1e-12)
    return bp, br, bf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", default="data/train_graphs.pkl")
    ap.add_argument("--val_graphs", default="data/validation_graphs.pkl")
    ap.add_argument("--test_graphs", default="data/test_graphs.pkl")

    ap.add_argument("--mol_ckpt", default="mol_gine_bert_clip.pt", help="Mol encoder checkpoint (state_dict).")
    ap.add_argument("--mol_node_dim", type=int, default=256)
    ap.add_argument("--mol_layers", type=int, default=4)
    ap.add_argument("--mol_out_dim", type=int, default=768)
    ap.add_argument("--mol_batch", type=int, default=256)

    ap.add_argument("--kedit", type=int, default=5)
    ap.add_argument("--use_feat", action="store_true")

    ap.add_argument("--t5_model", default="t5-base", help="t5-small, t5-base, t5-large, google/flan-t5-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_in", type=int, default=384)
    ap.add_argument("--max_out", type=int, default=128)

    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--length_penalty", type=float, default=0.8)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submission", default="submission_t5.csv")
    ap.add_argument("--save_ckpt", default="t5_editor_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    train_desc_by_id = load_descriptions_from_graphs(args.train_graphs)

    with open(args.val_graphs, "rb") as f:
        val_graphs = pickle.load(f)
    val_gt = {g.id: g.description for g in val_graphs}

    train_ds = PreprocessedGraphDataset(args.train_graphs)
    val_ds = PreprocessedGraphDataset(args.val_graphs)
    test_ds = PreprocessedGraphDataset(args.test_graphs)

    train_ids = train_ds.ids
    val_ids = val_ds.ids
    test_ids = test_ds.ids

    train_descs = [train_desc_by_id[i] for i in train_ids]
    val_targets = [val_gt[i] for i in val_ids]

    print(f"train/val/test sizes: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    if args.use_feat:
        with open(args.train_graphs, "rb") as f:
            train_graphs = pickle.load(f)
        with open(args.test_graphs, "rb") as f:
            test_graphs = pickle.load(f)

        train_id2g = {g.id: g for g in train_graphs}
        val_id2g = {g.id: g for g in val_graphs}
        test_id2g = {g.id: g for g in test_graphs}
    else:
        train_id2g = val_id2g = test_id2g = None

    if not os.path.exists(args.mol_ckpt):
        raise FileNotFoundError(
            f"Mol checkpoint not found: {args.mol_ckpt}\nProvide --mol_ckpt or train it first."
        )

    mol = MolGINE(node_dim=args.mol_node_dim, out_dim=args.mol_out_dim, layers=args.mol_layers).to(device)
    mol.load_state_dict(torch.load(args.mol_ckpt, map_location=device), strict=True)
    mol.eval()
    print("Loaded MolGINE:", args.mol_ckpt)

    train_ids_enc, train_mol_emb = encode_mols(mol, args.train_graphs, device, batch_size=args.mol_batch)
    val_ids_enc, val_mol_emb = encode_mols(mol, args.val_graphs, device, batch_size=args.mol_batch)
    test_ids_enc, test_mol_emb = encode_mols(mol, args.test_graphs, device, batch_size=args.mol_batch)

    assert train_ids_enc == train_ids
    assert val_ids_enc == val_ids
    assert test_ids_enc == test_ids

    k = args.kedit
    print("Retrieval Kedit =", k)

    sims_train = train_mol_emb @ train_mol_emb.t()
    sims_train.fill_diagonal_(-1e9)
    topk_train = sims_train.topk(k, dim=-1).indices

    sims_val = val_mol_emb @ train_mol_emb.t()
    topk_val = sims_val.topk(k, dim=-1).indices

    sims_test = test_mol_emb @ train_mol_emb.t()
    topk_test = sims_test.topk(k, dim=-1).indices

    train_inputs: List[str] = []
    train_targets: List[str] = []
    for i, tid in enumerate(tqdm(train_ids, desc="build T5 train inputs")):
        neigh = topk_train[i].tolist()
        neigh_descs = [train_descs[j] for j in neigh]
        feat = mol_feat(train_id2g[tid]) if args.use_feat else None
        train_inputs.append(build_prompt(neigh_descs, feat))
        train_targets.append(train_descs[i])

    val_inputs: List[str] = []
    for i, vid in enumerate(tqdm(val_ids, desc="build T5 val inputs")):
        neigh = topk_val[i].tolist()
        neigh_descs = [train_descs[j] for j in neigh]
        feat = mol_feat(val_id2g[vid]) if args.use_feat else None
        val_inputs.append(build_prompt(neigh_descs, feat))

    print("Loading T5:", args.t5_model)
    tok = AutoTokenizer.from_pretrained(args.t5_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model).to(device)

    train_dl = DataLoader(
        T5DS(train_inputs, train_targets),
        batch_size=args.batch,
        shuffle=True,
        collate_fn=make_collate_t5(tok, args.max_in, args.max_out)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_dl, desc=f"T5 train ep {ep}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        val_pred = generate_all(
            model, tok, val_inputs, device,
            batch_size=args.gen_batch,
            max_in=args.max_in,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
        bp, br, bf1 = bleu_f1(val_pred, val_targets)

        print("\n" + "=" * 80)
        print(f"Epoch {ep} | train_loss={float(np.mean(losses)):.4f}")
        print(f"BLEU precision: {bp:.4f}")
        print(f"BLEU recall:    {br:.4f}")
        print(f"BLEU F1:        {bf1:.4f}")
        print("=" * 80 + "\n")

        if bf1 > best_f1:
            best_f1 = bf1
            torch.save(model.state_dict(), args.save_ckpt)
            print(f"✓ saved best checkpoint -> {args.save_ckpt} (BLEU F1 {best_f1:.4f})\n")

    if os.path.exists(args.save_ckpt):
        model.load_state_dict(torch.load(args.save_ckpt, map_location=device))
        model.eval()

    test_inputs: List[str] = []
    for i, tid in enumerate(tqdm(test_ids, desc="build T5 test inputs")):
        neigh = topk_test[i].tolist()
        neigh_descs = [train_descs[j] for j in neigh]
        feat = mol_feat(test_id2g[tid]) if args.use_feat else None
        test_inputs.append(build_prompt(neigh_descs, feat))

    test_pred = generate_all(
        model, tok, test_inputs, device,
        batch_size=args.gen_batch,
        max_in=args.max_in,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )

    sub = pd.DataFrame({"ID": test_ids, "description": test_pred})
    sub.to_csv(args.submission, index=False)
    print(f"Saved submission -> {args.submission} ({len(sub)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
