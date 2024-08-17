
import torch
import sys
import os
import numpy as np
import pickle
import subprocess
import argparse
import shutil
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessor import generate_dset
from sparse_code import SparseCodeLayer, generate_dict
from input_output import MemmapCSR

def generate_dset_dict_codes(args):
    # Run preprocessing, dictionary learning, + sparse coding in a separate process to reduce RAM usage. Python 
    # garbage collector will hang onto GB of RAM from this initial processing. Child process gets its own 
    # Python interpreter + all memory is freed when that process ends.
    # NOTE: this function assumes no other memory maps have been used prior to this function running
    
    args_path = args.mmap_path + "/args.pkl"
    with open(args_path, "wb") as file: pickle.dump(args, file)
    
    # tell child where to write results
    dset_path = args.mmap_path + "/dset.pkl"
    phi_path = args.mmap_path + "/phi.pkl"
    label_path = args.mmap_path + "/label.pt"
    info_path = args.mmap_path + "/alphas_info.pkl"    

    # spawn child process using current Python executable, then wait for completion
    path = os.path.join(os.getcwd(), 'src')
    subprocess.run([f"{sys.executable}", "-c", f"import sys; sys.path.append('{path}'); import torch; torch.set_grad_enabled(False); from util import _new_process_main; _new_process_main('{args_path}', '{dset_path}', '{phi_path}', '{label_path}', '{info_path}')"])

    # read results from files
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        img_label = torch.load(label_path, weights_only=False)
        with open(phi_path, "rb") as file: phi = pickle.load(file)
        with open(dset_path, "rb") as file: dset = pickle.load(file)
        with open(info_path, "rb") as file: info = pickle.load(file)

    files = [f for f in os.listdir(args.mmap_path) if os.path.isfile(os.path.join(args.mmap_path, f)) and ".bin" in f]
    assert len(files) == 3, "Unexpected contents in mmap directory! Assumes only .bin files are for the memmory mapped sparse codes."

    def get_path(files, str):
        f = [i for i in files if str in i]
        assert len(f) == 1
        return "/" + f[0]

    shp = (int(info[0]), int(info[1]))
    data_dtype = np.dtype(info[2])
    idx_dtype = np.dtype(info[3])
    assert data_dtype == np.float32 and idx_dtype == np.int64
    data = np.memmap(args.mmap_path + get_path(files, 'data'), mode="r+", dtype=data_dtype)
    indptr = np.memmap(args.mmap_path + get_path(files, 'indptr'), mode="r+", dtype=idx_dtype)
    indices = np.memmap(args.mmap_path + get_path(files, 'indices'), mode="r+", dtype=idx_dtype)

    alphas = MemmapCSR((data, indices, indptr), shape=shp, dtype=data.dtype, copy=False)
    assert alphas.dtype == data_dtype

    # clean up
    os.remove(dset_path)
    os.remove(phi_path)
    os.remove(info_path)
    os.remove(label_path)
    os.remove(args_path)

    return dset, alphas, phi, img_label


def _new_process_main(args_path, dset_path, sc_path, label_path, info_path):
    # function run in the child process
    with torch.no_grad():
        with open(args_path, "rb") as file: args = pickle.load(file)

        # generate dataset, dictionary, and sparse codes
        dset = generate_dset(args)
        x, img_label = dset.generate_data(args.samples)
        phi, idx_list = generate_dict(args, x, args.dict_sz, args.dict_thresh)
        sc_layer = SparseCodeLayer(args.dict_sz, phi, args.gq_thresh, idx_list)
        alphas = sc_layer(args, x)


        # write result to files
        with open(sc_path, "wb") as file: pickle.dump(sc_layer, file)
        torch.save(img_label, label_path)
        with open(dset_path, "wb") as file: pickle.dump(dset, file)
        info = [f'{alphas.shape[0]}', f'{alphas.shape[1]}', f'{alphas.data.dtype}', f'{alphas.indptr.dtype}']
        with open(info_path, "wb") as file: pickle.dump(info, file)

        exit()


def validate_args(args):
    # Clear + create mmap directory
    assert args.mmap_path[-1] != '/', "Filename format error. Assumes no trailing slash."
    if os.path.exists(args.mmap_path):
        shutil.rmtree(args.mmap_path)
    os.mkdir(args.mmap_path)

    assert args.embed_dim <= args.dict_sz, f"Cannot have more embedding dimensions ({args.embed_dim}) than dictionary elements ({args.dict_sz})."
    if args.cov_chunk < 0: args.cov_chunk = args.dict_sz
    if args.inner_chunk < 0: args.inner_chunk = args.dict_sz
    
    assert args.samples > 0
    
    # account for horizontal augmentation
    args.samples *= 2
    return args


def generate_argparser():
    # Generic
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], help='dataset to use')
    parser.add_argument('--dataset-path', required=True, type=str, help='path to dataset (image datasets only)')
    parser.add_argument('--samples', default=50000, type=int, help='number of training samples to use')
    parser.add_argument('--test-samples', default=10000, type=int, help='number of training samples to use')
    parser.add_argument('--optim', default='two', choices=['one', 'two'], help='optimization equation to use from (2), naming follows the equation numbers from that paper. "one" is first deriv., "two" is second deriv.')

    # Image pre-processor
    parser.add_argument('--patch-sz', default=6, type=int, help='image patch size')
    parser.add_argument('--context-sz', default=3, type=int, help='other patches within this number of pixels is considered a neighbor')
    parser.add_argument('--grayscale_only', action='store_true', help='convert all input images to grayscale')
    parser.add_argument('--whiten_tol', default=1e-3, type=float, help='scaling of identity added before whitening')

    # Dictionary
    parser.add_argument('--dict-sz', default=300, type=int, help='sparse coding dictionary size. should be overcomplete, larger than input data dimension by around 10x')

    # Sparse-coding
    parser.add_argument('--gq_thresh', default=1, type=float, help='general sparse coding cosine similarity threshold. set to >= 1 to recover vector quantization')
    parser.add_argument('--dict_thresh', default=0.7, type=float, help='sparse coding dictionary element similarity threshold.')
    parser.add_argument('--zero_code_disable', action='store_true', help='ensure that no sparse codes are all zeros (map to nearest dict element if necessary)')
    parser.add_argument('--sc_chunk', default=50, type=int, help='chunk size used when calculating sparse coding')

    # SMT Embedding
    parser.add_argument('--embed-dim', default=384, type=int, help='feature manifold dimension (patch embedding dimension, image-level embedding will be much larger)')
    parser.add_argument('--disable_color_embed_drop', action='store_true', help='do not drop the first 16 embedding dim')

    # Classifier
    parser.add_argument('--nnclass-k', default=30, type=int, help='value of k for k-NN classifier')
    parser.add_argument('--knn_temp', default=0.03, type=float, help='temperatur for soft k-NN classifier')

    # Performance optimization
    parser.add_argument('--inner_chunk', default=8192, type=int, help='chunk size used when calculating the "inner" matrix for SMT optimization')
    parser.add_argument('--cov_chunk', default=512, type=int, help='chunk size used when calculating the "cov" matrix for SMT optimization')
    parser.add_argument('--cov_col_chunk', default=1000000, type=int, help='chunk size for breaking up columns in cov matrix calculation')       # 50k images --> 36450000 cols.
    
    parser.add_argument('--diff_op_a_chunk', default=750, type=int, help='alphas row chunk size used when applying differential operator to alphas')
    parser.add_argument('--diff_op_d_chunk', default=25, type=int, help='diff op column chunk size used when applying differential operator to alphas (in # of images)')

    parser.add_argument('--classify_chunk', default=50, type=int, help='chunk size used when classifying test-set images')
    parser.add_argument('--mmap-path', default="/tmp/smt-memmap", type=str, help='path to store temporary memory map files')
    parser.add_argument('--proj_row_chunk', default=32, type=int, help='SMT embedding projection rows batch size')
    parser.add_argument('--proj_col_chunk', default=500000, type=int, help='SMT embedding projection columns batch size')  
    parser.add_argument('--proj_cache_proc', default=2, type=int, help='SMT embedding projection matmul cache generation workers')

    return parser
