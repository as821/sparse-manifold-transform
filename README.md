# Sparse Manifold Transform (SMT)
Unofficial implementation of the Sparse Manifold Transform.

## Usage:

```
pip install -r requirements.txt
```

CIFAR10 (Top 1, Top 5) accuracies.

Small 1000 sample run: (60.55, 94.0)

```
python3 scripts/calc_smt.py --dataset=cifar10 --dataset-path=<DATASET_PATH> --ckpt-path=<CKPT_PATH> --embed-dim=384 --dict-sz=8192 --gq_thresh=0.3 --optim=two --samples=1000 --dict_thresh=0.5 --patch-sz=6
```
    
Full dataset: (76.94, 97.17)

```
python3 scripts/calc_smt.py --dataset=cifar10 --dataset-path=<DATASET PATH> --ckpt-path=<CKPT_PATH> --embed-dim=384 --dict-sz=8192 --gq_thresh=0.3 --optim=two --samples=50000 --dict_thresh=0.5 --patch-sz=6
```

Note that these results are lower than those presented in [2] (76.94% vs. 79.2%). This is likely due to a combination of differences in dictionary selection, unpublished hyperparameters such as the tolerance added to avoid singular whitening matrices, and the non-associativity of floating point operations which can highlight differences in implementation details.


## References:

[1]: The Sparse Manifold Transform https://arxiv.org/pdf/1806.08887.pdf 

[2]: Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform https://arxiv.org/pdf/2209.15261.pdf

