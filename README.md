# SADLE

Official model implementation for the manuscript:

**Subject-Adaptive Dual-Level Evidential Fusion of Multi-view Functional
Connectivity Networks for Brain Disease Identification**

This repository intentionally provides only the complete SADLE model. Dataset
preparation, training pipelines, experiment scripts, checkpoints, and clinical
data are not included.

## Requirements

- Python 3.10+
- PyTorch 2.0+

## Input and output

The model input is a resting-state fMRI time-series tensor with shape:

```text
[batch_size, time_points, num_brain_regions]
```

The forward pass returns six tensors:

1. PC-view Dirichlet parameters for every window;
2. HOFC-view Dirichlet parameters for every window;
3. SR-view Dirichlet parameters for every window;
4. MI-view Dirichlet parameters for every window;
5. metric-view fused Dirichlet parameters for every window;
6. final subject-level Dirichlet parameters after time-view fusion.

## Minimal example

```python
from argparse import Namespace

import torch

from model import SADLE


args = Namespace(
    num_features=116,
    n_classes=2,
    window_size=25,
    step_size=20,
    n_bins=2,
    alphas=1e-5,
    hidden_dim_pc=64,
    hidden_dim_hofc=64,
    hidden_dim_sr=64,
    hidden_dim_mi=64,
    mlp_hidden_pc=[64, 32],
    mlp_hidden_hofc=[64, 32],
    mlp_hidden_sr=[64, 32],
    mlp_hidden_mi=[64, 32],
    dropout_gcn_pc=0.3,
    dropout_gcn_hofc=0.3,
    dropout_gcn_sr=0.3,
    dropout_gcn_mi=0.3,
    dropout_mlp_pc=0.3,
    dropout_mlp_hofc=0.3,
    dropout_mlp_sr=0.3,
    dropout_mlp_mi=0.3,
)

model = SADLE(args)
x = torch.randn(2, 180, 116)
pc, hofc, sr, mi, window_fused, subject_fused = model(x)
prediction = subject_fused.argmax(dim=-1)
uncertainty = args.n_classes / subject_fused.sum(dim=-1)
```

The evidential training functions `evidence_loss` and `kl_divergence` are also
provided in `model.py`.

## Citation

Please cite the associated paper if this model is useful in your research.
