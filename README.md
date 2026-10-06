# E-PAGERec

Code for the submission "Enhancing Recommendation with Joint Clustering and
Graph-Based Representation Learning".

## Requirements

Python 3, PyTorch, NumPy and SciPy:

```
pip install -r requirements.txt
```

## Data

We use the Yelp2018, Gowalla and Amazon-Book splits released with LightGCN:
https://github.com/kuandeng/LightGCN/tree/master/Data

Put each dataset in its own folder with `train.txt` and `test.txt`, e.g. `data/yelp2018/`.
No preprocessing is needed. `epagerec.py` holds out 10% of each user's training
interactions for validation and builds the reduced training sets (`--train_frac`).
The test sets are the original ones.

## Running

The default values of `epagerec.py` are the settings of the paper.
A single run, with 20% of the training interactions:

```
python epagerec.py --data data/yelp2018 --out runs/example --train_frac 0.2
```

Each run writes `result.json` , `per_user.npz` (per-user NDCG@20), and `log.txt`.

`run_grid.py` runs the experiments of the paper. Replace the path and name for
Gowalla and Amazon-Book.

```
# Table 1 and Table 2: with and without communities, full data and 20%, 5 seeds
python run_grid.py --phase sparse_seeds --data data/yelp2018 --name yelp

# Fig. 2: other data fractions (Yelp2018)
python run_grid.py --phase sparse_seeds --data data/yelp2018 --name yelp --fracs 0.5,0.3,0.1

# Table 3: random communities, then the other variants (Yelp2018, 20%)
python run_grid.py --phase sparse_seeds --data data/yelp2018 --name yelp --modes random --fracs 0.2
python run_grid.py --phase ablation20 --data data/yelp2018 --name yelp

# number of communities K (Yelp2018, 20%)
python run_grid.py --phase sensitivity --data data/yelp2018 --name yelp

# Fig. 3: item embeddings and communities at epochs 3, 7, 15 and at the end of training
python run_grid.py --phase umap --data data/yelp2018 --name yelp
```


Small differences between repeated runs are expected because some parallel
operations are not deterministic.

## Baselines

All baselines use the same training, validation and test splits. The graph
collaborative filtering baselines and NCL were run with RecBole and RecBole-GNN with
their recommended hyperparameters. DCCF, BIGCF, LightCCF and FPSR were run with their
official implementations.
