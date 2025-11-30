# Hierarchical Molecular Language Models (HMLMs)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org)

## Overview

Hierarchical Molecular Language Models (HMLMs) represent a paradigm shift in computational systems biology by treating cellular signaling networks as specialized "molecular languages." This framework adapts transformer architectures to model complex biological signaling across multiple scales—from individual molecules to cellular responses, enabling superior predictive accuracy and mechanistic biological insights.

HMLMs achieve **30% improvement over Graph Neural Networks (GNNs)** and **52% improvement over Ordinary Differential Equation (ODE) models** in temporal signaling prediction, while maintaining robust performance under sparse temporal sampling conditions.

## Key Features

- **Multi-scale Integration**: Hierarchical processing of molecular, pathway, and cellular-level information
- **Graph-Structured Attention**: Adapted transformer mechanisms for network topology accommodation
- **Temporal Dynamics**: Explicit modeling of signaling kinetics across multiple timescales
- **Multi-modal Data Fusion**: Integration of phosphoproteomics, transcriptomics, imaging, and perturbation data
- **Interpretable Attention Mechanisms**: Identifies biologically meaningful cross-talk patterns and regulatory relationships
- **Sparse Data Handling**: Superior performance with limited temporal sampling (MSE = 0.041 with only 4 timepoints)

## Performance Highlights

| Temporal Resolution | HMLM | GNN | ODE | LDE | Bayesian Networks |
|--------|------|-----|-----|-----|-------------------|
| 4 timepoints (sparse) | **0.041** | 0.054 | 0.121 | 0.071 | 0.071 |
| 8 timepoints (medium) | **0.050** | 0.075 | 0.120 | 0.091 | 0.091 |
| 16 timepoints (high) | **0.058** | 0.083 | 0.121 | 0.096 | 0.095 |

**Improvements over baseline methods:**
- **24% improvement** over GNN (4 timepoints: 0.041 vs 0.054 MSE)
- **66% improvement** over ODE (4 timepoints: 0.041 vs 0.121 MSE)
- **30% improvement** over GNN at full resolution (16 timepoints)
- **52% improvement** over ODE at full resolution (16 timepoints)
- **Superior sparse data handling**: Maintains robust performance with minimal timepoints

## Installation

### Requirements
- Python 3.8 or higher
- PyTorch 1.9+
- NumPy 1.19+
- Pandas 1.1+
- Scikit-learn 0.24+
- NetworkX 2.5+

### Setup

```bash
# Clone the repository
git clone https://github.com/HasiHays/HMLMs.git
cd HMLMs

# Create virtual environment (recommended)
python -m venv hmlm_env
source hmlm_env/bin/activate  # On Windows: hmlm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from hmlm import HMLM, CardiacFibroblastNetwork

# Initialize network
network = CardiacFibroblastNetwork()

# Create and train model
model = HMLM(
    network_topology=network.graph,
    num_scales=3,
    attention_heads=8,
    hidden_dim=512
)

# Train on signaling data
model.train(
    temporal_data=signaling_data,
    num_epochs=50,
    learning_rate=0.001
)

# Predict cellular responses
predictions = model.predict(
    initial_conditions=conditions,
    time_steps=100
)

# Extract biological insights
attention_patterns = model.analyze_attention()
cross_talk = model.identify_pathway_crosstalk()
```

## Usage Examples

### Basic Model Training

See `HMLMs_main.ipynb` for comprehensive examples including:
- Data preprocessing and normalization
- Multi-scale embedding generation
- Model training and validation
- Attention mechanism visualization

### Cardiac Fibroblast Signaling Prediction

```python
# Define experimental conditions
conditions = {
    'control': {'TGF_beta': 0, 'mechanical_strain': 0},
    'TGF_beta': {'TGF_beta': 1.0, 'mechanical_strain': 0},
    'strain': {'TGF_beta': 0, 'mechanical_strain': 1.0},
    'combined': {'TGF_beta': 1.0, 'mechanical_strain': 1.0}
}

# Predict dynamics across conditions
for condition_name, stim in conditions.items():
    dynamics = model.predict_dynamics(stim, time_steps=100)
    # Analyze fibrosis markers, contractility, etc.
```

### Attention-Based Pathway Analysis

```python
# Extract molecular-scale attention weights
mol_attention = model.get_molecular_attention()

# Identify high-confidence regulatory edges
significant_edges = mol_attention[mol_attention > 0.7]

# Pathway-level cross-talk quantification
pathway_crosstalk = model.quantify_cross_talk()
print(f"TGF-β to MAPK cross-talk: {pathway_crosstalk['TGFb_MAPK']:.3f}")
```

## Architecture

### Core Components

1. **Information Transducers**: Fundamental units modeling biological entities (proteins, complexes, pathways)
   - Input Space: External signals and stimuli
   - State Space: Internal configurations (conformational, activation, binding, modification, localization states)
   - Output Space: Signal transformations and functional responses

2. **Graph-Structured Attention**: Adapted transformer mechanisms respecting network topology
   - Node embeddings incorporating entity type, features, and graph positional information
   - Multi-head attention operating within network neighborhoods
   - Temporal attention capturing signal propagation delays

3. **Scale-Bridging Operators**:
   - **Aggregation (Ω↑)**: Combines molecular-level information into pathway-level representations
   - **Decomposition (Ω↓)**: Distributes pathway signals to constituent molecules
   - **Translation (Ω↔)**: Converts information between representational formats

4. **Hierarchical Integration**:
   - Within-scale attention: Local interactions at each biological scale
   - Cross-scale attention: Information flow across scales
   - Feed-forward networks: Non-linear integration

## Data Requirements

### Input Data Formats

The model accepts multi-modal biological data:

- **Phosphoproteomics**: Protein phosphorylation time series (measured via mass spectrometry or immunoassays)
- **Transcriptomics**: Gene expression profiles (RNA-seq or qPCR)
- **Imaging Data**: Subcellular localization, morphological features (immunofluorescence or live-cell imaging)
- **Perturbation Data**: Effects of knockdowns, inhibitors, or genetic modifications

### Example Data Structure

```python
# Temporal signaling data
data = {
    'time_points': np.array([0, 1, 5, 10, 30, 60, 120]),  # minutes
    'molecules': {
        'ERK_phos': np.array([0.0, 0.1, 0.8, 0.9, 0.5, 0.2, 0.0]),
        'AKT_phos': np.array([0.0, 0.05, 0.3, 0.5, 0.7, 0.8, 0.9]),
        'p53': np.array([1.0, 1.0, 1.5, 2.0, 1.8, 1.2, 1.0])
    },
    'conditions': 'TGF_beta_stimulation'
}
```

## Model Evaluation

The repository includes comprehensive evaluation tools:

```python
from hmlm.evaluation import evaluate_model, plot_predictions

# Evaluate on test data
metrics = evaluate_model(
    model=trained_model,
    test_data=test_signaling_data,
    metrics=['mse', 'pearson_correlation', 'temporal_resolution']
)

# Visualize predictions vs. observations
plot_predictions(
    model=trained_model,
    data=test_data,
    molecules=['ERK', 'AKT', 'p38'],
    conditions=['control', 'TGF_beta']
)
```

## Mathematical Foundation

### Information Transducer Definition

Each biological entity is modeled as an information transducer **T = (X, Y, S, f, g)**:

- **State Transition**: s(t+1) = f(x(t), s(t))
- **Output Function**: y(t) = g(s(t))

### Graph Attention Mechanism

```
GraphAttention_v(Q, K, V) = softmax(Q_v K^T_{N(v)} / √d_k) V_{N(v)}
```

where N(v) represents the neighborhood of vertex v in the signaling network.

### Temporal Embedding

```
h^(0)_v(t) = h̃^(0)_v + τ(t)
```

Combines static node embeddings with learned temporal patterns across multiple timescales.

## Publications & Citation

If you use HMLMs in your research, please cite: (Pre-print available at: arXiv:submit/7034338 [cs.AI] 30 Nov 2025)

```bibtex
@article{Hays2025HMLM,
  title={Hierarchical Molecular Language Models: Bridging Molecular Mechanisms with Cellular Phenotypes via AI-Driven Molecular Language Representation},
  author={Hays, Hasi and Yu, Yue and Richardson, William},
  journal={[Journal Name]},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

## Support & Issues

For bug reports, feature requests, or questions:
- **GitHub Issues**: [Create an issue](https://github.com/HasiHays/HMLMs/issues)
- **Email**: hasih@uark.edu

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This work was supported by:
- National Institutes of Health (NIGMS R01GM157589)
- Department of Defense (DEPSCoR FA9550-22-1-0379)

---

**Last Updated**: November 2025  
**Maintained By**: Hasi Hays (hasih@uark.edu)