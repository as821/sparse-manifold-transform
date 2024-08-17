# Sparse Manifold Transform (SMT)
Unofficial implementation of the Sparse Manifold Transform.


## References:

(1): The Sparse Manifold Transform https://arxiv.org/pdf/1806.08887.pdf 

(2): Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform https://arxiv.org/pdf/2209.15261.pdf


## Usage: (OUT OF DATE!)
#### Replicate unit disk spherical harmonics (Ref. 1 Figure 2):

  `cd src; python3 main.py --dataset=circle --embed-dim=21 --dict-sz=300`

#### MNIST

   `python3 src/main.py --dataset=mnist --dataset-path=<PATH> --gq_thresh=1 --embed-dim=16 --dict-sz=16384 --dict-path=<DICT-PATH> --gpu --dict-batch=50000 --dict-mx-iter=500`

#### CIFAR10

   `python3 src/main.py --dataset=cifar10 --dataset-path=<PATH> --embed-dim=32 --dict-sz=8192 --dict-path=<DICT-PATH> --gq_thresh=0.95 --mmap-path=<path to tmp or a mounted SSD>`

  - additional options optimized for ~25gb RAM and 24gb VRAM (>1 TB of disk recommended):

    `--multi_process --gpu --sc_chunk_sz=50000 --matmul_chunk=500000 --cov_chunk=500 --inner_chunk=45`

#### MISC
- For image datasets, the dictionary only needs to be generated once, after which it can be loaded from the given dictionary path using the `--load-dict` flag.
- Useful OS settings
  - Set [/proc/sys/vm/overcommit_memory](https://linux.die.net/man/5/proc) to 1 to allow the [creation of large memory mapped files](https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type).
  - Due to heavy reliance on memory mapping and swapping pages to/from disk, the OOM daemon may kill the program due to "memory pressure" despite having large amounts of memory available. See [here](https://www.freedesktop.org/software/systemd/man/latest/oomd.conf.html) and [here](https://www.freedesktop.org/software/systemd/man/latest/systemd-oomd.service.html#) for reference. Practically, I had to set SwapUsedLimit, DefaultMemoryPressureLimit, DefaultMemoryPressureDurationSecs in `/usr/lib/systemd/oomd.conf.d/*-oomd-default.conf` in addition to ManagedOOMMemoryPressure=auto and ManagedOOMMemoryPressureLimit in `/usr/lib/systemd/system/user@.service.d/*-oomd-user-service-defaults.conf` in order to see results.
  - Increase swap file size, optionally mess with swappiness settings.


## Implementation Details

#### Processing Pipeline
1) conversion of training images into patches and preprocessing
2) dictionary generation
3) sparse coding of image patches in the dictionary basis
4) embedding optimization
5) test set preprocessing, embedding, and classification

#### Optimizations
- Use of sparse matrix data structures, backed by memory mapped files to reduce memory usage and take advantage of sparsity inherent in the method.
- Partial construction of the differential operator D since it can be applied to all patches from each image separately.
- Batched GPU implementation of vector quantization (1-sparse coding) and general k-sparse coding.
- Use of convolutions + GPU for conversion of images to patches, calculation of contextual means.
- GPU acceleration of k-Means during dictionary learning
- Run image preprocessing, dictionary learning, and sparse coding in a separate process as a cheap form of memory management. (Python garbage collection cannot handle cyclic dependencies)



