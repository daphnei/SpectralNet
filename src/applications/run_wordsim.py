'''
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
'''

import sys, os
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

from core.data import get_wordsim_data
from siamesenet import run_net

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        'model_in': 'best_512_128.model',           # where to read a pre-trained model from, or None to initialize from scratch
        # 'model_in': None,           # where to read a pre-trained model from, or None to initialize from scratch
        # 'model_out': 'test_noimages_nonlinear_larger_300d.model'          # where to write the trained model to, or None to not write it out
        'model_out': None,
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
wordsim_params = {
    'n_clusters': 1,
    'use_code_space': False,             # enable / disable code space embedding
    'affinity': 'siamese',              # affinity type: siamese / knn
    'n_nbrs': 3,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
    'scale_nbr': 3,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
                                        # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
                                        # sampled from the datset

    'siam_k': 3,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
                                        # a 'positive' pair by siamese net

    'siam_ne': 200,                     # number of training epochs for siamese net
    'siam_lr': 1e-3,                    # initial learning rate for siamese net
    'spec_lr': 1e-3,                    # initial learning rate for spectral net
    'siam_patience': 5,                 # early stopping patience for siamese net
    'siam_drop': 0.8,                   # learning rate scheduler decay for siamese net
    'batch_size': 1024,                 # batch size for spectral net
    'siam_reg': 1e-2,                   # regularization parameter for siamese net
    'spec_reg': 5e-1,                   # regularization parameter for spectral net
    'siam_n': None,                     # subset of the dataset used to construct training pairs for siamese net
    'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
    'arch': [                           # network architecture. if different architectures are desired for siamese net and
                                        #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
        {'type': 'relu', 'size': 512},
        # {'type': 'relu', 'size': 256},
        # {'type': 'relu', 'size': 512},
        {'type': 'relu', 'size': 128},
        ],
    'use_approx': True,                # enable / disable approximate nearest neighbors
    'use_all_data': False,             # enable to use all data for training (no test set)
    'emb_dim': 300,                    # the dimension of the word embeddings
    'use_extra_distances': False,      # incorporate extra distances (such as image distance) into loss
    'use_word2vec_features': True,     # Use word2vec as input features 
    'use_image_features': True,        # Use Image mean/standard deviations as input features
    'normalize_features': False,       # Normalize input features.
}
params.update(wordsim_params)

# LOAD DATA
data = get_wordsim_data(params)

# RUN EXPERIMENT
x_spectralnet, y_spectralnet = run_net(data, params)

if args.dset in ['cc', 'cc_semisup']:
    # run plotting script
    import plot_2d
    plot_2d.process(x_spectralnet, y_spectralnet, data, params)
