'''
spectralnet.py: contains run function for spectralnet
'''
import sys, os, pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi
import sklearn.linear_model

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop

from core import train
from core import costs
from core import networks
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds

def run_net(data, params):
    #
    # UNPACK DATA
    #

    pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']
    extra_distances_train = data['siamese']['differences'][0]
    extra_distances_val = data['siamese']['differences'][1]

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = [pairs_train.shape[2]]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }


    extra_distances = Input(shape=[1],name='ExtraDistances')

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    # run only if we are using a siamese network
    siamese_net = networks.SiameseNet(inputs, extra_distances, y_true, params)
    if params['model_in'] is None:
        print('Training...')
        history = siamese_net.train(
                pairs_train, dist_train, extra_distances_train,
                pairs_val, dist_val, extra_distances_val,
                params['siam_lr'], params['siam_drop'], params['siam_patience'],
                params['siam_ne'], params['siam_batch_size'])
        print("finished training")
        if params['model_out'] is not None:
            siamese_net.net.save_weights(params['model_out'])
    else:
        # siamese_net.net.load_weights(params['model_in'])
        print('Loaded weights from checkpoint.')

    #
    # EVALUATE
    #

    all_x_sim = []
    all_z_sim = []
    all_y = []
    for idx in range(0, pairs_val.shape[0]):

      x = pairs_val[idx, :, :]
      
      z = siamese_net.predict(x, batch_sizes)

      x_sim = np.linalg.norm(x[0,:] - x[1,:])
      z_sim = np.linalg.norm(z[0,:] - z[1,:])

      all_x_sim.append(x_sim)
      all_z_sim.append(z_sim)
      all_y.append(dist_val[idx])

      # orig_sim = np.linalg.norm(x[0,:] - x[1,:])
  
    model1 = sklearn.linear_model.LogisticRegression()
    model1.fit(np.array(all_x_sim).reshape(-1, 1), np.array(all_y))
    score = model1.score(np.array(all_x_sim).reshape(-1, 1), np.array(all_y))
    print('Score with original space: ' + str(score))

    score = np.sum(np.array(all_y) == 1) / float(len(all_y))
    print('Score with predict 1: ' + str(score))

    score = np.sum(np.array(all_y) == 0) / float(len(all_y))
    print('Score with predict 0: ' + str(score))

    preds = np.array(all_z_sim) > 0.09
    preds = preds.astype(int)
    score = np.sum(np.array(all_y) == preds) / float(len(all_y))
    print('Score with predict z > 0.09 as 1: ' + str(score))

    preds = np.array(all_z_sim) > 1
    preds = preds.astype(int)
    score = np.sum(np.array(all_y) == preds) / float(len(all_y))
    print('Score with predict z > 1 as 1: ' + str(score))

    model2 = sklearn.linear_model.LogisticRegression()
    model2.fit(np.array(all_z_sim).reshape(-1, 1), np.array(all_y))
    score = model2.score(np.array(all_z_sim).reshape(-1, 1), np.array(all_y))
    print('Score with transformed space: ' + str(score))

    # xx = np.mgrid[0:0.15:0.005].reshape(-1, 1)
    # probs = model2.predict_proba(xx)[:, 1].reshape(xx.shape)
      
    plt.figure(1)
    plt.scatter(all_x_sim, all_z_sim, s=10, c=all_y) 
    # for i in range(len(probs)):
      # plt.axhline(y=xx[i])    

    plt.show()
