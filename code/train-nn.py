"""
Train a Neural Network to predict visits to malicious sites
"""

import argparse
import csv
import os

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    RobustScaler, QuantileTransformer, Normalizer
from Imbalance import *

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, Dense, Activation, \
    Flatten, Lambda, normalization
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument('--outdir', help='where to store neural network?')
    parser.add_argument('--features_file', help='where are the features stored?')
    parser.add_argument('--features_val_file', help='where are the validation features stored?')
    parser.add_argument('--train_min_time', type=int, help='the first date to to pick training data from (to allow building some history)')
    parser.add_argument('--train_max_time', type=int, help='the latest date to pick training data from')
    parser.add_argument('--val_max_time', type=int, help='the latest date to pick validation data from')
    parser.add_argument('--min_accesses', type=int, default=10, help='min number of accesses before we start predicting (session needs to build history)')
    parser.add_argument('--delta_max', type=int, default=60, help='how long into the future do we want to predict (in seconds)')
    parser.add_argument('--aknn_full', type=str, \
                        help='path to precomputed AnnoyIndex (see: aknn-preprocess.py; used w. SMOTE/ADASYN)--computed over the entire training set')
    parser.add_argument('--aknn_positive', type=str, \
                        help='path to preprocessed AnnoyIndex (see: aknn-preprocess.py; used w. ADASYN)--computed over the positive/rare examples in training')
    parser.add_argument('--imbalance_alg', type=int, default=0, help='algorithm to solve imbalance: '\
                        '0- None, 1- SMOTE, 2- ADASYN')
    parser.add_argument('--oversample_times', type=float, default=1, help='for each positive example, generate ''oversample_times'' samples')
    parser.add_argument('--k_nn', type=int, default=5, help='K for K-Nearest Neighbor operations')
    parser.add_argument('--feature_colnames_file', default='./feature-colnames.txt', help='file containing the column names of features')
    parser.add_argument('--feature_types', type=parse_feature_types, default='SPC',\
                        help='Types of features to use. S: Self-reported. P: Past behavior. C: Contextual.')
    parser.add_argument('--keep_access_prob', type=float, default=1.,\
                        help='Probability of using an HTTP request/access in training')
    parser.add_argument('--scaler_type', type=str, default='minmax',\
                        help='type of feature scaler used (standard/minmax/robust/quantile/quantile_gauss/normalizer)')
    parser.add_argument('--nn_arch', type=str, default='cnn', help='type of neural net: mlp or cnn?')
    return parser.parse_args()


def parse_feature_types(in_arg):
    in_arg = str(in_arg).upper()
    for c in in_arg:
        if not c in 'SPC':
            raise ValueError('Feature type can only include the letters \'C\', \'P\', or \'S\'. ')
    return in_arg

def solve_imbalance(imbalance_alg, X, oversample_times, k, aknn_full, aknn_positive):
    if imbalance_alg==0:
        return X
    if imbalance_alg==1:
        X = SMOTE(X, k, oversample_times, aknn_positive)
    if imbalance_alg==2:
        X = ADASYN(X, k, oversample_times, aknn_full, aknn_positive)
    return X

# model parameters
n_epochs = 50
batch_size = 128
n_hidden1 = 25
n_hidden2 = 25
n_hidden3 = 2
lr = 5e-5
fw = 5
nf = 128
nl = 3
b1 = 0.9
b2 = 0.99
weight_decay = 1e-5
eps = 1e-8
pos_frac = 0.5 # "positive fraction": fraction of positive examples in minibatches

# use to select data samples and label them
args = parse_args()
train_min_time = args.train_min_time
train_max_time = args.train_max_time
val_max_time   = args.val_max_time
delta_max = args.delta_max
min_accesses = args.min_accesses
data_file = args.features_file
val_file = args.features_val_file
model_store_path = args.outdir
aknn_full = args.aknn_full
aknn_positive = args.aknn_positive
oversample_times = args.oversample_times
imbalance_alg = args.imbalance_alg
k_nn = args.k_nn
feature_colnames_file = args.feature_colnames_file
feature_types = args.feature_types

# from feature types, figure the the columns (i.e., field names) to keep
fin = open(feature_colnames_file, mode='r')
all_feature_colnames = fin.readlines()
feature_colnames = []
for colname in all_feature_colnames:
    if 'S' in feature_types and 'SelfReported' in colname:
        feature_colnames.append(colname[:-1])
    if 'P' in feature_types and 'PastBehavior' in colname:
        feature_colnames.append(colname[:-1])
    if 'C' in feature_types and 'Contextual' in colname:
        feature_colnames.append(colname[:-1])

# read train data
X0_tr = []
X1_tr = []
fin = open(data_file, mode='r')
reader = csv.DictReader(fin, delimiter='\t')
for row in reader:
    start_time = int(row['communication_start_time'])
    if row['mal'] == '0':
        session_label = 0
    elif int(row['from_mal_sec'])>=0:
        # make sure we're predicting before visit to malicious website
        continue
    elif int(row['to_mal_sec'])>delta_max:
        # granular control over predictions---only closer to the exposure
        # we have enough information to tell that we're in an exposed
        # browsing session
        continue
    else:
        session_label = 1
    if start_time<=train_min_time:
        # we're still building history about participants
        continue
    if start_time>train_max_time:
        # the sample is past training time
        continue
    if int(row['Contextual:session_len_access'])<min_accesses:
        # don't try to predict if it's too early in the session
        continue
    if np.random.rand()>args.keep_access_prob:
        # uniformly pick acceesses for training with p=keep_access_prob
        continue
    features = [float(row[colname]) for colname in feature_colnames ]
    # training sample
    if session_label==0:
        X0_tr.append(features)
    else:
        X1_tr.append(features)
fin.close()

# read validation data
fin2 = open(val_file, mode='r')
reader = csv.DictReader(fin2, delimiter='\t')
X0_val = []
X1_val = []
for row in reader:
    start_time = int(row['communication_start_time'])
    if row['mal'] == '0':
        session_label = 0
    elif int(row['from_mal_sec'])>=0: 
        # make sure we're predicting before visit to malicious website
        continue
    elif int(row['to_mal_sec'])>delta_max:
        # granular control over predictions---only closer to the exposure
        # we have enough information to tell that we're in an exposed
        # browsing session
        continue
    else:
        session_label = 1
    if start_time<=train_max_time:
        # too early to be in validation set
        continue
    if start_time>val_max_time:
        # sample is past the validation period
        continue
    if int(row['Contextual:session_len_access'])<min_accesses:
        # don't try to predict if it's too early in the session
        continue
    if np.random.rand()>args.keep_access_prob:
        # uniformly pick acceesses for validation with p=keep_access_prob
        continue
    features = [float(row[colname]) for colname in feature_colnames]
    # validation sample
    if session_label==0:
        X0_val.append(features)
    else:
        X1_val.append(features)
fin2.close()

# Switch to np arrays
X0_tr = np.asarray(X0_tr, dtype=np.float)
X1_tr = np.asarray(X1_tr, dtype=np.float)
X0_val = np.asarray(X0_val, dtype=np.float)
X1_val = np.asarray(X1_val, dtype=np.float)

# normalize data
scaler_type = args.scaler_type
if scaler_type=='standard':
    scaler = StandardScaler()
elif scaler_type=='minmax':
    scaler = MinMaxScaler()
elif scaler_type=='robust':
    scaler = RobustScaler(quantile_range=(25, 75))
elif scaler_type=='quantile':
   scaler = QuantileTransformer(output_distribution='uniform')
elif scaler_type=='quantile_gauss':
    scaler = QuantileTransformer(output_distribution='normal')
elif scaler_type=='normalizer':
    scaler = Normalizer()
else:
    ValueError('Unknown scaler_type \'%s\''%(scaler_type))
scaler.fit(np.concatenate((X0_tr, X1_tr), axis=0))
X0_tr = scaler.transform(X0_tr)
X1_tr = scaler.transform(X1_tr)
X0_val = scaler.transform(X0_val)
X1_val = scaler.transform(X1_val)

# Run alg. to deal with imbalance issues
X1_tr = solve_imbalance(imbalance_alg, X1_tr, oversample_times, k_nn, aknn_full, aknn_positive)

# initialize and compile NN model
dim_features = len(feature_colnames)
if args.nn_arch=='mlp':
    model = Sequential()
    model.add(Dense(units=n_hidden1, input_dim=dim_features))
    model.add(Activation('relu'))
    model.add(Dense(units=n_hidden2))
    model.add(Activation('relu'))
    model.add(Dense(units=n_hidden3))
    model.add(Activation('softmax'))
elif args.nn_arch=='cnn':
    inputs = Input(shape=(dim_features,))
    outputs = Lambda(lambda x: keras.backend.expand_dims(x, axis=2))(inputs)
    for l in range(nl):
        outputs = Conv1D(nf, fw)(outputs)
        outputs = Activation('relu')(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation('softmax')(outputs)
    model = Model(inputs, outputs)
model.compile(loss=keras.losses.categorical_crossentropy,\
              optimizer=keras.optimizers.Adam(lr=lr, beta_1=b1, beta_2=b2, decay=weight_decay))

# start training
n_X1 = X1_tr.shape[0]
n_X0 = X0_tr.shape[0]
n_val = min(X0_val.shape[0], X1_val.shape[0])
step_size1 = int(pos_frac*batch_size)
step_size0 = batch_size - step_size1
for epoch in range(n_epochs):
    # shuffle train data
    np.random.shuffle(X0_tr)
    np.random.shuffle(X1_tr)
    # train
    first_idx1 = 0
    first_idx0 = 0
    while first_idx1<n_X1 and first_idx0<n_X0:
        last_idx1 = min([first_idx1+step_size1, n_X1])
        last_idx0 = min([first_idx0+step_size0, n_X0])
        X1_batch = X1_tr[first_idx1:last_idx1]
        X0_batch = X0_tr[first_idx0:last_idx0]
        X_batch = np.concatenate((X0_batch, X1_batch))
        sz1 = last_idx1 - first_idx1
        sz0 = last_idx0 - first_idx0
        Y_batch = np.zeros((sz0+sz1,2))
        Y_batch[:sz0,0] = 1
        Y_batch[sz0:,1] = 1
        model.train_on_batch(X_batch, Y_batch)
        first_idx1 = last_idx1
        first_idx0 = last_idx0
    # validate
    if epoch%5==0:
        top1_err = 0
        fp = 0
        tp = 0
        np.random.shuffle(X0_val)
        np.random.shuffle(X1_val)
        predictions = model.predict(X0_val[:n_val], batch_size=batch_size)
        for i in range(predictions.shape[0]):
            if predictions[i][0]<=predictions[i][1]:
                top1_err += 1
                fp += 1
        predictions = model.predict(X1_val[:n_val], batch_size=batch_size)
        for i in range(predictions.shape[0]):
            if predictions[i][1]<predictions[i][0]:
                top1_err += 1
            else:
                tp += 1
        top1_err = float(top1_err)/(2*X1_val.shape[0])
        tpr = float(tp) / n_val
        fpr = float(fp) / n_val
        print 'top1_err = %0.4f, tpr = %0.4f, fpr = %0.4f'%(top1_err,tpr,fpr)

# store model and normalization data
model.save(model_store_path+args.nn_arch+feature_types+'-features-' +\
           scaler_type + '-' + str(train_min_time) + 'TO' + str(train_max_time) +'.h5')
joblib.dump(scaler, model_store_path+scaler_type+'-'+feature_types+'-features'+\
            str(train_min_time) + 'TO' + str(train_max_time) + '.dump')

