# Molecule-Text Retrieval

Graph neural network for molecular graph and text description retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Place your preprocessed graph data files in the `data/` directory:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

## Usage

Run the following scripts in order:

### 1. Inspect Graph Data

Check the structure and contents of your graph files:

```bash
python inspect_graph_data.py
```

### 2. Generate Description Embeddings

Create BERT embeddings for molecular descriptions:

```bash
python generate_description_embeddings.py
```

This generates:
- `data/train_embeddings.csv`
- `data/validation_embeddings.csv`

### 3. Train GCN Model

Train the graph neural network:

```bash
python train_gcn.py
```

This creates `model_checkpoint.pt`.

### 4. Run Retrieval

Retrieve descriptions for test molecules:

```bash
python retrieval_answer.py
```

This generates `test_retrieved_descriptions.csv` with retrieved descriptions for each test molecule.

## Output

- `model_checkpoint.pt`: Trained GCN model
- `test_retrieved_descriptions.csv`: Retrieved descriptions for test set

