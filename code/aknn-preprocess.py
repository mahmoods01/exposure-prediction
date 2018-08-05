"""
Compute and store the AnnoyIndex which is used for
fast Approximate K-NN (AKNN) computations.

Only needed if SMOTE/ADASYN are used while training
the Neural Networks.
"""

import argparse
import csv
import os
from os import path

import numpy as np
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    RobustScaler, QuantileTransformer, Normalizer

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument('--features_file', help='where are the features stored?')
    parser.add_argument('--train_min_time', type=int, help='the first date to to pick training data from (to allow building some history)')
    parser.add_argument('--train_max_time', type=int, help='the latest date to pick training data from')
    parser.add_argument('--val_max_time', type=int, help='the latest date to pick validation data from')
    parser.add_argument('--min_accesses', type=int, help='min number of accesses before we start predicting (session needs to build history)')
    parser.add_argument('--delta_max', type=int, help='how long into the future do we want to predict (in seconds)')
    parser.add_argument('--scaler_type', type=str, default='minmax',\
                        help='type of feature scaler used (standard/minmax/robust/quantile/quantile_gauss/normalizer)')
    parser.add_argument('--outdir', type=str, help='where to store the output')
    return parser.parse_args()

# some consts
dim_features = 274
eps = 1e-8

# use to select data samples and label them
args = parse_args()
train_min_time = args.train_min_time
train_max_time = args.train_max_time
val_max_time   = args.val_max_time
delta_max = args.delta_max
min_accesses = args.min_accesses
data_file = args.features_file

# read train and val data
X0_tr = []
X1_tr = []
fin = open(data_file, mode='r')
line_cnt = 0
reader = csv.DictReader(fin, delimiter='\t')
for row in reader:
    line_cnt += 1
    if line_cnt % 1000000 == 0:
        print datetime.now().strftime('%m/%d %H:%M:%S') + '  ' + str(line_cnt / 1000000) + 'M loaded'
    start_time = int(row['communication_start_time'])
    if row['mal'] == '0':
        session_label = 0
    else:
        session_label = 1
    if start_time<=train_min_time:
        # we're still bulding history about participants
        continue
    if start_time>val_max_time:
        # done picking training and validation data
        continue
    if int(row['Contextual:session_len_access'])<min_accesses:
        # don't try to predict if it's too early in the session
        continue
    features = [float(v) for v in [row[field] for field in reader.fieldnames[4:]] ]
    if start_time<=train_max_time:
        # training sample
        if session_label==0:
            X0_tr.append(features)
        else:
            X1_tr.append(features)

# Switch to np arrays
X0_tr = np.asarray(X0_tr, dtype=np.float)
X1_tr = np.asarray(X1_tr, dtype=np.float)

# Normalize data
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

# Compute AnnoyIndex-es
t = AnnoyIndex(dim_features)
X = np.concatenate((X1_tr, X0_tr))
for i in range(X.shape[0]):
    v = X[i]
    t.add_item(i, v)
t.build(10)
t.save(path.join(args.outdir, 'annoy-full.bin'))
t = AnnoyIndex(dim_features)
for i in range(X1_tr.shape[0]):
    v = X1_tr[i]
    t.add_item(i, v)
t.build(10)
t.save(path.join(args.outdir, 'annoy-positive.bin'))

