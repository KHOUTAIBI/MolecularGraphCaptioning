#!/usr/bin/env python3
"""
Auto-generated from train_improved_bert_t5(1).ipynb
Runs end-to-end: trains MolGINE (BERT-emb alignment), builds retrieval prompts, trains GPT-2 editor (optionally LoRA), evaluates BLEU/BERTScore per epoch, and writes submission CSV.

Notes:
- Expects data/{train,validation,test}_graphs.pkl
- Expects data_utils.py in the same directory (or PYTHONPATH).
- If checkpoints exist, the script will reuse them.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')


# =====================
# Notebook cell 0
# =====================

import os, pickle, re, math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool

import sacrebleu
from bert_score import score as bertscore

from data_utils import PreprocessedGraphDataset, collate_fn, load_descriptions_from_graphs, x_map, e_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

with open(VAL_GRAPHS, "rb") as f:
    val_graphs = pickle.load(f)
    
val_gt = {g.id: g.description for g in val_graphs}

# =====================
# Notebook cell 1
# =====================

train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS)
val_ds   = PreprocessedGraphDataset(VAL_GRAPHS)
test_ds  = PreprocessedGraphDataset(TEST_GRAPHS)

train_ids = train_ds.ids
val_ids   = val_ds.ids
test_ids  = test_ds.ids

train_desc_by_id = load_descriptions_from_graphs(TRAIN_GRAPHS)
train_descs = [train_desc_by_id[i] for i in train_ids]

val_ref = [val_gt[i] for i in val_ids]

print(len(train_ids), len(val_ids), len(test_ids))
print("train cap sample:", train_descs[0][:120])
print("val ref sample:", val_ref[0][:120])

# =====================
# Notebook cell 2
# =====================

BERT_NAME = "bert-base-uncased"
MAX_TOKEN_LENGTH = 128

bert_tok = AutoTokenizer.from_pretrained(BERT_NAME)
bert = AutoModel.from_pretrained(BERT_NAME).to(DEVICE)
bert.eval()

@torch.no_grad()
def encode_texts_bert(texts, batch_size=64):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT encode"):
        batch = texts[i:i+batch_size]
        inp = bert_tok(batch, return_tensors="pt", truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
        inp = {k:v.to(DEVICE) for k,v in inp.items()}
        h = bert(**inp).last_hidden_state[:,0,:]   # CLS
        out.append(h.float().cpu())
    embs = torch.cat(out, 0)
    return F.normalize(embs, dim=-1)

train_text_emb = encode_texts_bert(train_descs, batch_size=64)
print("train_text_emb:", train_text_emb.shape)

# =====================
# Notebook cell 3
# =====================

class CatFeatureEncoder(nn.Module):
    def __init__(self, sizes, emb_dim):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, emb_dim) for s in sizes])

    def forward(self, x_cat):
        out = 0
        for j, emb in enumerate(self.embs):
            out = out + emb(x_cat[:, j])
        return out

class MolGINE(nn.Module):
    def __init__(self, node_dim=256, out_dim=768, layers=4):
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

    def forward(self, batch: Batch):
        x = self.node_enc(batch.x.long())
        e = self.edge_enc(batch.edge_attr.long())
        for conv in self.convs:
            x = conv(x, batch.edge_index, e)
            x = F.relu(x)
        g = global_add_pool(x, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)

# =====================
# Notebook cell 4
# =====================

class GraphTextEmbDS(Dataset):
    def __init__(self, graphs_pkl, id_list, id2idx, text_emb):
        self.graphs_pkl = graphs_pkl
        self.ds = PreprocessedGraphDataset(graphs_pkl)
        self.ids = self.ds.ids
        self.id2idx = id2idx
        self.text_emb = text_emb

    def __len__(self): return len(self.ds)

    def __getitem__(self, i):
        # PreprocessedGraphDataset supports indexing to return graph
        g = self.ds[i]
        tid = self.ids[i]
        t = self.text_emb[self.id2idx[tid]]
        return g, t

def collate_graph_text(batch):
    graphs, texts = zip(*batch)
    batch_graphs = Batch.from_data_list(list(graphs))
    text = torch.stack(list(texts), 0)
    return batch_graphs, text
    

train_id2idx = {tid:i for i, tid in enumerate(train_ids)}

train_gt_ds = GraphTextEmbDS(TRAIN_GRAPHS, train_ids, train_id2idx, train_text_emb)
train_dl = DataLoader(train_gt_ds, batch_size=64, shuffle=True, collate_fn=collate_graph_text)

# =====================
# Notebook cell 5
# =====================

EPOCHS_MOL = 50

# =====================
# Notebook cell 6
# =====================

def clip_infonce(g, t, tau=0.07):
    # g,t: [B,D], normalized
    logits = (t @ g.t()) / tau
    labels = torch.arange(g.size(0), device=g.device)
    loss_t2g = F.cross_entropy(logits, labels)
    loss_g2t = F.cross_entropy(logits.t(), labels)
    return 0.5*(loss_t2g + loss_g2t)

mol = MolGINE(node_dim=256, out_dim=train_text_emb.size(1), layers=4).to(DEVICE)
opt = torch.optim.AdamW(mol.parameters(), lr=2e-4, weight_decay=1e-2)

for ep in range(1, EPOCHS_MOL + 1):  # 5 epochs baseline
    mol.train()
    losses = []
    for graphs, t in tqdm(train_dl, desc=f"mol train ep {ep}"):
        
        graphs = graphs.to(DEVICE)
        t = F.normalize(t.to(DEVICE), dim=-1)
        g = mol(graphs)
        loss = clip_infonce(g, t, tau=0.07)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    print(f"ep {ep} loss={np.mean(losses):.4f}")

torch.save(mol.state_dict(), "mol_gine_bert_clip.pt")
print("saved mol_gine_bert_clip.pt")

# =====================
# Notebook cell 8
# =====================

def bleu_f1(preds, refs):
    bleu_prec = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    bleu_rec  = sacrebleu.corpus_bleu(refs, [preds]).score / 100.0
    f1 = 2 * bleu_prec * bleu_rec / (bleu_prec + bleu_rec + 1e-12)
    return bleu_prec, bleu_rec, f1

def bert_f1(preds, refs, batch_size=16):
    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        model_type=CHEMBERTA,
        num_layers=6,      # critical fix
        device=DEVICE,
        batch_size=batch_size,
        verbose=False
    )
    return float(P.mean()), float(R.mean()), float(F1.mean())

# =====================
# Notebook cell 9
# =====================

@torch.no_grad()
def encode_mols(graphs_pkl, batch_size=256):
    ds = PreprocessedGraphDataset(graphs_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_g = []
    for graphs in tqdm(dl, desc=f"encode mol {graphs_pkl}"):
        graphs = graphs.to(DEVICE)
        all_g.append(mol(graphs).cpu())
    return ds.ids, F.normalize(torch.cat(all_g, 0), dim=-1)

mol.load_state_dict(torch.load("mol_gine_bert_clip.pt", map_location=DEVICE), strict=True)
mol.eval()

train_ids_enc, train_mol_emb = encode_mols(TRAIN_GRAPHS)
val_ids_enc,   val_mol_emb   = encode_mols(VAL_GRAPHS)
test_ids_enc,  test_mol_emb  = encode_mols(TEST_GRAPHS)

assert train_ids_enc == train_ids
assert val_ids_enc == val_ids
assert test_ids_enc == test_ids

# =====================
# Notebook cell 10
# =====================

K = 5  # for editor prompt
sims_val = val_mol_emb @ train_mol_emb.t()
topk_val = sims_val.topk(K, dim=-1).indices  # [Nval,K]

sims_test = test_mol_emb @ train_mol_emb.t()
topk_test = sims_test.topk(K, dim=-1).indices

# =====================
# Notebook cell 11
# =====================

x_keys = list(x_map.keys())
atomic_col = x_keys.index("atomic_num") if "atomic_num" in x_keys else 0

inv_atomic = None
try:
    inv_atomic = {idx:int(val) for val,idx in x_map["atomic_num"].items()}
except:
    pass

COMMON = [6,7,8,9,15,16,17,35,53]  # C N O F P S Cl Br I

with open(TRAIN_GRAPHS,"rb") as f: train_graphs = pickle.load(f)
with open(TEST_GRAPHS,"rb") as f: test_graphs  = pickle.load(f)

train_id2g = {g.id:g for g in train_graphs}
test_id2g  = {g.id:g for g in test_graphs}

def mol_feat(graph):
    x = graph.x
    n_atoms = x.size(0)
    n_bonds = graph.edge_index.size(1)//2
    z_cat = x[:, atomic_col].long().cpu().numpy()
    if inv_atomic is not None:
        z = np.array([inv_atomic.get(int(c), -1) for c in z_cat])
    else:
        z = z_cat
    counts = {Z:int((z==Z).sum()) for Z in COMMON}
    counts_str = " ".join([f"{Z}:{counts[Z]}" for Z in COMMON if counts[Z]>0])
    return f"atoms {n_atoms} bonds {n_bonds} atomsZ {counts_str if counts_str else 'none'}"

# =====================
# Notebook cell 12
# =====================

def build_prompt(neigh_descs, feat):
    # Keep prompt short + consistent
    lines = ["NEIGHBORS:"]
    for i,d in enumerate(neigh_descs,1):
        lines.append(f"{i}) {d}")
    lines.append(f"FEATS: {feat}")
    lines.append("CAPTION:")
    return "\n".join(lines)

# =====================
# Notebook cell 13
# =====================

Kedit = 5
sims_train = train_mol_emb @ train_mol_emb.t()
sims_train.fill_diagonal_(-1e9)
topk_train = sims_train.topk(Kedit, dim=-1).indices

train_prompts = []
train_targets = []

for i, tid in enumerate(tqdm(train_ids, desc="build editor train")):
    neigh = topk_train[i].tolist()
    neigh_descs = [train_descs[j] for j in neigh]
    feat = mol_feat(train_id2g[tid])
    prompt = build_prompt(neigh_descs, feat)
    train_prompts.append(prompt)
    train_targets.append(train_id2g[tid].description)

# =====================
# Notebook cell 14
# =====================

# Build validation prompts (for BLEU / BERTScore evaluation)

with open(VAL_GRAPHS, "rb") as f:
    val_graphs = pickle.load(f)

val_id2g = {g.id: g for g in val_graphs}
print("val_id2g size:", len(val_id2g), "sample key:", next(iter(val_id2g)))


Kedit = 5  # must match training

sims_val = val_mol_emb @ train_mol_emb.t()
topk_val = sims_val.topk(Kedit, dim=-1).indices  # [Nval, K]

val_prompts = []
val_targets = []   # ground-truth val captions

for i, vid in enumerate(tqdm(val_ids, desc="build editor val")):
    neigh = topk_val[i].tolist()
    neigh_descs = [train_descs[j] for j in neigh]

    # IMPORTANT:
    # use val graph features, NOT train
    feat = mol_feat(val_id2g[vid])

    prompt = build_prompt(neigh_descs, feat)
    val_prompts.append(prompt)
    val_targets.append(val_gt[vid])

# =====================
# Notebook cell 15
# =====================

print(val_prompts[0][:300])
print("GT:", val_targets[0][:200])

# =====================
# Notebook cell 16
# =====================

GPT_NAME = "gpt2"

gpt_tok = AutoTokenizer.from_pretrained(GPT_NAME)
gpt_tok.pad_token = gpt_tok.eos_token  # critical for batching

gpt = AutoModelForCausalLM.from_pretrained(GPT_NAME).to(DEVICE)

# =====================
# Notebook cell 17
# =====================

class GPTEditorDS(Dataset):     
    def __init__(self, prompts, targets, max_len=320):         
        self.prompts = prompts         
        self.targets = targets         
        self.max_len = max_len      
        
    def __getitem__(self, i):         
        prompt = self.prompts[i]         
        target = self.targets[i]          
        # Full text         
        full = prompt + " " + target + gpt_tok.eos_token          
        # tokenize full         
        enc = gpt_tok(full, truncation=True, max_length=self.max_len, padding=False)         
        input_ids = enc["input_ids"]          
        # tokenize prompt alone (same truncation!)         
        prompt_ids = gpt_tok(prompt, truncation=True, max_length=self.max_len, padding=False)["input_ids"]         
        Lp = len(prompt_ids)          
        labels = input_ids.copy()         
        labels[:Lp] = [-100] * Lp          
        # SAFETY: if everything is masked, force at least last token to be supervised         
        if all(x == -100 for x in labels):             
            labels[-1] = input_ids[-1]          
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        

    def __len__(self):          
        return len(self.prompts)      

def collate_gpt(batch):
    input_ids, labels = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=gpt_tok.pad_token_id)
    labels    = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attn = (input_ids != gpt_tok.pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# =====================
# Notebook cell 18
# =====================

from peft import LoraConfig, get_peft_model, TaskType

GPT_NAME = "gpt2"
gpt_tok = AutoTokenizer.from_pretrained(GPT_NAME)
gpt_tok.pad_token = gpt_tok.eos_token

base_gpt = AutoModelForCausalLM.from_pretrained(GPT_NAME).to(DEVICE)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
)

gpt = get_peft_model(base_gpt, lora_cfg)
gpt.print_trainable_parameters()

# =====================
# Notebook cell 19
# =====================

def generate_caption(prompt, max_new=96):
    gpt.eval()
    enc = gpt_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=320
    ).to(DEVICE)

    out = gpt.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
        eos_token_id=gpt_tok.eos_token_id,
        pad_token_id=gpt_tok.pad_token_id,
    )
    text = gpt_tok.decode(out[0], skip_special_tokens=True)
    # keep only the part after "CAPTION:"
    if "CAPTION:" in text:
        text = text.split("CAPTION:", 1)[1].strip()
    return text

# =====================
# Notebook cell 20
# =====================

gpt_train_ds = GPTEditorDS(train_prompts, train_targets, max_len=320)
gpt_train_dl = DataLoader(gpt_train_ds, batch_size=4, shuffle=True, collate_fn=collate_gpt)  # <= reduce if OOM

opt = torch.optim.AdamW(gpt.parameters(), lr=3e-5)

# =====================
# Notebook cell 21
# =====================

def bleu_f1(preds, refs):
    bleu_prec = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0
    bleu_rec  = sacrebleu.corpus_bleu(refs, [preds]).score / 100.0
    f1 = 2 * bleu_prec * bleu_rec / (bleu_prec + bleu_rec + 1e-12)
    return bleu_prec, bleu_rec, f1


EPOCHS = 10   # adjust
BEST_BLEU = -1.0

for ep in range(1, EPOCHS + 1):

    # =========================
    # TRAIN
    # =========================
    gpt.train()
    train_losses = []

    for batch in tqdm(gpt_train_dl, desc=f"GPT2 train epoch {ep}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = gpt(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            print("Non-finite loss detected. Skipping batch.")
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        opt.step()

        train_losses.append(loss.item())

    print(f"\nEpoch {ep} | Train loss: {np.mean(train_losses):.4f}")

    # =========================
    # VALIDATION GENERATION
    # =========================
    gpt.eval()
    val_pred = []
    

    with torch.no_grad():
        for prompt in tqdm(val_prompts, desc=f"GPT2 val gen epoch {ep}"):
            val_pred.append(generate_caption(prompt))

    # =========================
    # METRICS
    # =========================
    bleu = sacrebleu.corpus_bleu(val_pred, [val_ref]).score / 100.0
    bp, br, bf1 = bleu_f1(val_pred, val_ref)

    print("-" * 80)
    print(f"Epoch {ep} VALIDATION METRICS")
    print(f"BLEU (corpus): {bleu:.4f}")
    print(f"BLEU precision: {bp:.4f}")
    print(f"BLEU recall:    {br:.4f}")
    print(f"BLEU F1:        {bf1:.4f}")

    # OPTIONAL: BERTScore (comment out if slow)
    P, R, F1 = bertscore(
        cands=val_pred,
        refs=val_ref,
        model_type=BERT_NAME,
        num_layers=6,
        device=DEVICE,
        batch_size=16,
        verbose=False
    )

    print(f"BERTScore P:  {float(P.mean()):.4f}")
    print(f"BERTScore R:  {float(R.mean()):.4f}")
    print(f"BERTScore F1: {float(F1.mean()):.4f}")
    print("-" * 80)

    # =========================
    # SAVE BEST
    # =========================
    if bf1 > BEST_BLEU:
        BEST_BLEU = bf1
        torch.save(gpt.state_dict(), "gpt2_editor_best.pt")
        print("Saved best model (BLEU-F1 improved)")

# =====================
# Notebook cell 22
# =====================

tiny_rerank.eval()
with torch.no_grad():
    pick = tiny_rerank(X.to(DEVICE)).argmax(dim=1).cpu().numpy()

val_pred_bleu_rerank = []
for i in range(len(val_ids)):
    cand = topk_idx_val[i].tolist()
    val_pred_bleu_rerank.append(train_descs[cand[pick[i]]])

bp, br, bf1 = bleu_f1(val_pred_bleu_rerank, val_ref)
p, r, f1 = bert_f1(val_pred_bleu_rerank, val_ref, batch_size=16)
print("BLEU-aware TinyRerank | BLEU_F1:", bf1, "| BERT F1:", f1)

# =====================
# Notebook cell 23
# =====================

K = 50
sims_test = test_mol_emb @ train_text_emb.t()
topk_idx_test = sims_test.topk(K, dim=-1).indices
chosen_test = medoid_pick(topk_idx_test, train_text_emb)
test_pred_medoid = [train_descs[j] for j in chosen_test]

sub = pd.DataFrame({"ID": test_ids, "description": test_pred_medoid})
sub.to_csv("submission_medoid.csv", index=False)
print("saved submission_medoid.csv", sub.shape)

# =====================
# Notebook cell 24
# =====================

# build Xtest: [Ntest,K,3]
Xtest_list = []
for i in tqdm(range(len(test_ids)), desc="build Xtest"):
    cand = topk_idx_test[i].tolist()
    sem = (test_mol_emb[i:i+1] @ train_text_emb[cand].t()).squeeze(0).cpu().numpy()
    lf  = train_len[cand]
    lm  = train_lm[cand]
    Xtest_list.append(np.stack([sem, lf, lm], axis=1))

Xtest = torch.tensor(np.stack(Xtest_list, 0), dtype=torch.float32)

tiny_rerank.eval()
with torch.no_grad():
    pick_test = tiny_rerank(Xtest.to(DEVICE)).argmax(dim=1).cpu().numpy()

test_pred_rerank = []
for i in range(len(test_ids)):
    cand = topk_idx_test[i].tolist()
    test_pred_rerank.append(train_descs[cand[pick_test[i]]])

sub = pd.DataFrame({"ID": test_ids, "description": test_pred_rerank})
sub.to_csv("submission_rerank.csv", index=False)
print("saved submission_rerank.csv", sub.shape)

# # =====================
# # Notebook cell 25
# # =====================

# Kedit = 5

# # Train neighbors (exclude self)
# sims_train = train_mol_emb @ train_mol_emb.t()
# sims_train.fill_diagonal_(-1e9)
# topk_train = sims_train.topk(Kedit, dim=-1).indices

# # Val neighbors
# sims_val2 = val_mol_emb @ train_mol_emb.t()
# topk_val2 = sims_val2.topk(Kedit, dim=-1).indices

# # Test neighbors
# sims_test2 = test_mol_emb @ train_mol_emb.t()
# topk_test2 = sims_test2.topk(Kedit, dim=-1).indices

# # =====================
# # Notebook cell 26
# # =====================

# def load_graphs(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# train_graphs = load_graphs(TRAIN_GRAPHS)
# test_graphs  = load_graphs(TEST_GRAPHS)

# train_id2g = {g.id: g for g in train_graphs}
# test_id2g  = {g.id: g for g in test_graphs}

# def mol_features_str(graph):
#     x = graph.x
#     n_atoms = x.size(0)
#     n_bonds = graph.edge_index.size(1) // 2
#     aromatic = int(x[:, -2].sum().item()) if x.size(1) >= 2 else 0
#     ring     = int(x[:, -1].sum().item()) if x.size(1) >= 1 else 0
#     return f"atoms={n_atoms}; bonds={n_bonds}; aromatic={aromatic}; ring={ring}"

# def build_editor_input(neigh_descs, feat):
#     lines = ["NEIGHBORS:"]
#     for i, d in enumerate(neigh_descs, 1):
#         lines.append(f"{i}) {d}")
#     lines.append(f"MOL: {feat}")
#     lines.append("TASK: rewrite a correct caption for this molecule.")
#     return "\n".join(lines)

# # =====================
# # Notebook cell 27
# # =====================

# # targets = GT captions
# train_targets = [train_id2g[tid].description for tid in train_ids]
# val_targets   = [val_gt[vid] for vid in val_ids]

# train_inputs = []
# for i, tid in enumerate(tqdm(train_ids, desc="editor train inputs")):
#     feat = mol_features_str(train_id2g[tid])
#     neigh = topk_train[i].tolist()
#     neigh_descs = [train_descs[j] for j in neigh]
#     train_inputs.append(build_editor_input(neigh_descs, feat))

# val_inputs = []
# for i, vid in enumerate(tqdm(val_ids, desc="editor val inputs")):
#     # val graphs are in validation pkl, use val_gt + approximate features via train_id2g not possible.
#     # simplest: reuse text-only neighbors + omit features OR load val graphs if you want features.
#     # We'll omit features here to keep it robust:
#     neigh = topk_val2[i].tolist()
#     neigh_descs = [train_descs[j] for j in neigh]
#     val_inputs.append(build_editor_input(neigh_descs, feat="unknown"))

# # =====================
# # Notebook cell 28
# # =====================

# MODEL_NAME = "t5-small"
# t5_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# class EditorDS(torch.utils.data.Dataset):
#     def __init__(self, inputs, targets):
#         self.inputs = inputs
#         self.targets = targets
#     def __len__(self): return len(self.inputs)
#     def __getitem__(self, i): return self.inputs[i], self.targets[i]

# def collate_editor(batch, max_in=384, max_out=128):
#     inp, tgt = zip(*batch)
#     enc = t5_tok(list(inp), padding=True, truncation=True, max_length=max_in, return_tensors="pt")
#     with t5_tok.as_target_tokenizer():
#         dec = t5_tok(list(tgt), padding=True, truncation=True, max_length=max_out, return_tensors="pt")
#     labels = dec["input_ids"]
#     labels[labels == t5_tok.pad_token_id] = -100
#     enc["labels"] = labels
#     return enc

# train_dl = DataLoader(EditorDS(train_inputs, train_targets), batch_size=8, shuffle=True, collate_fn=collate_editor)
# val_dl   = DataLoader(EditorDS(val_inputs,   val_targets),   batch_size=8, shuffle=False, collate_fn=collate_editor)

# opt = torch.optim.AdamW(t5.parameters(), lr=3e-4)

# def gen_preds(model, inputs, batch=8):
#     model.eval()
#     preds = []
#     for i in tqdm(range(0, len(inputs), batch), desc="generate"):
#         b = inputs[i:i+batch]
#         enc = t5_tok(b, padding=True, truncation=True, max_length=384, return_tensors="pt")
#         enc = {k: v.to(DEVICE) for k, v in enc.items()}
#         out = model.generate(**enc, max_new_tokens=96, num_beams=4)
#         preds += t5_tok.batch_decode(out, skip_special_tokens=True)
#     return preds

# best_bleu = -1
# for ep in range(1, 4):  # 3 epochs
#     t5.train()
#     losses = []
#     for batch in tqdm(train_dl, desc=f"t5 train ep {ep}"):
#         batch = {k: v.to(DEVICE) for k, v in batch.items()}
#         loss = t5(**batch).loss
#         opt.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(t5.parameters(), 1.0)
#         opt.step()
#         losses.append(loss.item())

#     val_preds = gen_preds(t5, val_inputs, batch=8)
#     bleu = sacrebleu.corpus_bleu(val_preds, [val_targets]).score / 100.0
#     print(f"ep {ep} loss={np.mean(losses):.4f} val_bleu={bleu:.4f}")
#     if bleu > best_bleu:
#         best_bleu = bleu
#         torch.save(t5.state_dict(), "t5_editor.pt")
#         print("saved best -> t5_editor.pt")

# # =====================
# # Notebook cell 29
# # =====================

# t5.load_state_dict(torch.load("t5_editor.pt", map_location=DEVICE))
# t5.eval()

# test_inputs = []
# for i, tid in enumerate(tqdm(test_ids, desc="editor test inputs")):
#     feat = mol_features_str(test_id2g[tid])
#     neigh = topk_test2[i].tolist()
#     neigh_descs = [train_descs[j] for j in neigh]
#     test_inputs.append(build_editor_input(neigh_descs, feat))

# test_preds = gen_preds(t5, test_inputs, batch=8)

# sub = pd.DataFrame({"ID": test_ids, "description": test_preds})
# sub.to_csv("submission_editor.csv", index=False)
# print("saved submission_editor.csv", sub.shape)

# # =====================
# # Notebook cell 30
# # =====================

# sub.head()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_mol_train", action="store_true", help="Skip MolGINE training if checkpoint exists.")
    p.add_argument("--skip_gpt_train", action="store_true", help="Skip GPT-2 editor training if checkpoint exists.")
    p.add_argument("--mol_ckpt", default="mol_gine_bert_clip.pt")
    p.add_argument("--gpt_ckpt", default="gpt2_editor_best.pt")
    p.add_argument("--submission", default="submission.csv")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    # Make args visible to the notebook-converted code if it references it.
    globals()["ARGS"] = args
