# compile C files into shared library. call from top-level directory of repository
OPTIM_OPTIONS="-O3 -fno-math-errno -fno-trapping-math -march=znver3 -mtune=znver3"

NVCC_OPTIONS="-c -g -I./src/c/src -Wno-deprecated-declarations -o"
nvcc $NVCC_OPTIONS src/c/bin/util.o src/c/src/util.c 
nvcc $NVCC_OPTIONS src/c/bin/file_wrapper.o src/c/src/file_wrapper.c -Xcompiler="-fPIC $OPTIM_OPTIONS"
nvcc $NVCC_OPTIONS src/c/bin/csr_postproc.o src/c/src/cuda/csr_postproc.c -Xcompiler="-fPIC $OPTIM_OPTIONS"

nvcc $NVCC_OPTIONS src/c/bin/pinned_circ_buffer.o src/c/src/cuda/pinned_circ_buffer.c -Xcompiler="-pthread -fPIC $OPTIM_OPTIONS"
nvcc $NVCC_OPTIONS src/c/bin/cuda_util.o src/c/src/cuda/cuda_util.c -D_GNU_SOURCE
nvcc $NVCC_OPTIONS src/c/bin/spmm.o src/c/src/cuda/spmm.c
nvcc $NVCC_OPTIONS src/c/bin/spgemm_dense_res.o src/c/src/cuda/spgemm_dense_res.c
nvcc $NVCC_OPTIONS src/c/bin/spgemm_sparse_res.o src/c/src/cuda/spgemm_sparse_res.c -Xcompiler="-pthread -fPIC $OPTIM_OPTIONS"
nvcc $NVCC_OPTIONS src/c/bin/sync.o src/c/src/sync.c -Xcompiler="-pthread -fPIC $OPTIM_OPTIONS"
nvcc -shared -g -lcusparse -lcublas -Xcompiler="-pthread $OPTIM_OPTIONS" -o src/c/bin/cuda_c_func.so src/c/bin/spgemm_dense_res.o src/c/bin/spmm.o src/c/bin/cuda_util.o src/c/bin/spgemm_sparse_res.o src/c/bin/util.o src/c/bin/file_wrapper.o src/c/bin/sync.o src/c/bin/csr_postproc.o src/c/bin/pinned_circ_buffer.o
rm -f src/c/bin/*.o

GCC_OPTIONS="-c -g -I./src/c/src $OPTIM_OPTIONS -o"
gcc $GCC_OPTIONS src/c/bin/util.o src/c/src/util.c -D_GNU_SOURCE
gcc -fPIC $GCC_OPTIONS src/c/bin/coo2csr.o src/c/src/sparse_utils/coo2csr.c -D_GNU_SOURCE
gcc -shared -g $OPTIM_OPTIONS -o src/c/bin/coo2csr.so src/c/bin/coo2csr.o src/c/bin/util.o

gcc $GCC_OPTIONS src/c/bin/file_wrapper.o src/c/src/file_wrapper.c
gcc -fPIC -pthread $GCC_OPTIONS src/c/bin/csr_column_slice.o src/c/src/sparse_utils/csr_column_slice.c -D_GNU_SOURCE
gcc -shared -g $OPTIM_OPTIONS -o src/c/bin/csr_column_slice.so src/c/bin/file_wrapper.o src/c/bin/csr_column_slice.o src/c/bin/util.o
rm -f src/c/bin/*.o
