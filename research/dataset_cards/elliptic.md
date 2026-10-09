# Dataset card: Elliptic Bitcoin transaction graph

| Item | Value |
|---|---|
| Source | Weber et al. 2019 (Elliptic / MIT-IBM Watson AI Lab). Mirror used: Hugging Face `yhoma/elliptic-bitcoin-dataset` (tag `license:mit`) |
| Downloaded | 203,769 transaction nodes, 234,355 edges, 49 time steps, 166 features; 4,545 illicit / 42,019 licit labelled / 157,205 unknown |
| Licence | the original Elliptic release is on Kaggle; terms should be checked upstream - the mirror's MIT tag is not authoritative |
| Supports | graph construction, temporal modelling (49 timesteps), illicit-flow detection |
| Does NOT support | account-level mule labels, APP labels, devices |

Used as an independent temporal-graph benchmark (train time steps 1-34, test 35-49 per standard protocol). Not APP fraud.
