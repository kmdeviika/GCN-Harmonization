# GCN-Based EEG Spatial Harmonization

Reference implementation of the GCN electrode-harmonization model described in:

> K M *et al* 2026 *J. Neural Eng.* https://doi.org/10.1088/1741-2552/ae9344

If you use this code, please cite the paper above.


## Requirements

- Python 3.10
- [Modal](https://modal.com) account and CLI configured (`modal token new`)
- Dependencies listed in `requirements.txt` (installed automatically inside the Modal image)

## Data

Place the following inside `dataset_59_gcn/`:
- `S01_EEG_MI.mat` – `S07_EEG_MI.mat` (BCI Competition IV Dataset 1, 59 channels)
- `oostenveld_sphere2_59_reference_coordinates.csv` (real spherical electrode coordinates)


## Citation

```bibtex
@article{KM2026,
  author  = {K M and others},
  title   = {},
  journal = {Journal of Neural Engineering},
  year    = {2026},
  doi     = {10.1088/1741-2552/ae9344},
  url     = {https://doi.org/10.1088/1741-2552/ae9344}
}
```
