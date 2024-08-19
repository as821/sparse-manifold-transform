# Sparse Manifold Transform (SMT)
Unofficial implementation of the Sparse Manifold Transform.

## Installation

Install the required dependencies through the provided `requirements.txt`.

Using provided C implementation of various sparse matrix operations is strongly recommended for best performance. Assumes gcc and nvcc are installed and in your PATH.
```
mkdir src/c/bin
./src/c/build.sh
```

SMT is very memory intensive and when run on the full CIFAR10 dataset it operates on multiple 100+ GB matrices at time. This implementation relies on memory mapping and disk caching to keep memory usage under control. Adjusting the following Linux settings is recommended.

  - Set [/proc/sys/vm/overcommit_memory](https://linux.die.net/man/5/proc) to 1 to allow the [creation of large memory mapped files](https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type).
  - Due to heavy reliance on memory mapping and swapping pages to/from disk, the OOM daemon may kill the program due to "memory pressure" despite having large amounts of memory available. See [here](https://www.freedesktop.org/software/systemd/man/latest/oomd.conf.html) and [here](https://www.freedesktop.org/software/systemd/man/latest/systemd-oomd.service.html#) for reference. Practically, I had to set SwapUsedLimit, DefaultMemoryPressureLimit, DefaultMemoryPressureDurationSecs in `/usr/lib/systemd/oomd.conf.d/*-oomd-default.conf` in addition to ManagedOOMMemoryPressure=auto and ManagedOOMMemoryPressureLimit in `/usr/lib/systemd/system/user@.service.d/*-oomd-user-service-defaults.conf` in order to see results.

The Sparse Manifold Transform can use 1-2TB of disk space when run on the full CIFAR10 dataset and its performance relies heavily on the abilty to read and write quickly to storage. Use of a dedicated NVME SSD (vs. an HDD or a boot drive) is recommended. An NVIDIA GPU is not required, but is critical for performance on large runs. 

When calculating the covariance and the final cost matrices, 100s of GB will be rapidly read from the written drive, onto the GPU, and the results will be written back. This can quickly cause thermal throttling of the SSD. [Active cooling solutions](https://www.amazon.com/gp/product/B08Y8GC4DF/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) work best despite looking ridiculous. The impact of SMT on your drive's endurance and temperature can be inspected with `smartctl`. 


## Usage:

CIFAR10 Results are given as (Top 1, Top 5) accuracy. Options are optimized for 128GB RAM and 24GB VRAM.

Small 1000 sample run: (58.25, 93.11)

```
python3 scripts/calc_smt.py --dataset=cifar10 --dataset-path=/home/astange/slots/data/cifar --embed-dim=384 --dict-sz=8192 --mmap-path=/ssd1/smt-mmap --gq_thresh=0.3 --optim=two --samples=1000 --context-sz=32 --dict_thresh=0.38 --classify_chunk=500 --patch-sz=6 --diff_op_a_chunk=1024 --diff_op_d_chunk=100  --cov_chunk=1024
```
    
Full dataset: (78.06, 97.84)

```
python3 scripts/calc_smt.py --dataset=cifar10 --dataset-path=/home/astange/slots/data/cifar --embed-dim=384 --dict-sz=8192 --mmap-path=/ssd1/smt-mmap --gq_thresh=0.3 --optim=two --samples=50000 --context-sz=32 --dict_thresh=0.35 --classify_chunk=500 --patch-sz=6 --diff_op_a_chunk=1024 --diff_op_d_chunk=100  --cov_chunk=1024
```

Note that these results are slightly lower than those presented in [2] (78.06 % vs. 79.2%). This is likely due to a combination of differences in dictionary selection, unpublished hyperparameters such as the tolerance added to avoid singular whitening matrices, and the fact that the non-associativity of floating point operations can inflate the impact of differences in implementation details. 

## Implementation Details

### Repository Organization
The Python files provided in `src` implement the vast majority of the application logic. The `src/c` directory contains optimized subroutines for sparse matrix format conversions, memory-efficient column-slicing of CSR matrices, and sparse-sparse and sparse-dense matrix multiplications through the cuSPARSE API. A basic set of tests for the correctness of these C implementations is provided in `src/c/test`. 

### Rough Processing Pipeline
1) conversion of training images into patches, context mean removal, and whitening
2) "random" dictionary generation
3) representation of image patches in the dictionary basis
4) embedding optimization
5) test set preprocessing, embedding, and classification


## References:

[1]: The Sparse Manifold Transform https://arxiv.org/pdf/1806.08887.pdf 

[2]: Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform https://arxiv.org/pdf/2209.15261.pdf

