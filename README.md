# GCN-based EEG Spatial Harmonization

This repository provides the GCN harmonization implementation 

## Architecture

Each EEG trial is represented as:

```text
X: [N_c, N_s]
```

where `N_c` is the number of electrode nodes and `N_s` is the time-series
embedding length of each node.



## Graph construction

Edge weights are computed from pairwise geodesic distances on a spherical head
model with:

- head radius: 10 cm;
- Gaussian sigma: 5 cm;
- sparsity threshold: 0.001;
- self-loops;
- symmetric degree normalization.

## Training

During self-supervised training, complete electrode embeddings are randomly
masked. The two-layer GCN reconstructs the complete EEG graph. The training
objective is the mean squared error over all electrodes and all time samples,


Training and validation subjects are separated using a subject-wise split.

## Harmonization

Target EEG is aligned to the common 59-channel reference montage. Recorded
electrodes are placed in their matching positions and unavailable reference
electrodes are zero-filled before GCN inference. The GCN outputs a complete
59-channel reconstructed EEG graph.


