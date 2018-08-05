"""
Evaluate a trained neural network on data collected after the 
training/validation data
"""

import argparse
import csv
import os

import numpy as np
import keras
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import auc


from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    RobustScaler, QuantileTransformer, Normalizer

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument('--model_path', help='where to find neural network')
    parser.add_argument('--normalization_path', help='where to find the values used for normalization')
    parser.add_argument('--features_file', help='where are the features stored?')
    parser.add_argument('--test_min_time', type=int, help='the first date to pick test data from')
    parser.add_argument('--test_max_time', type=int, help='the last date to pick test data from')
    parser.add_argument('--min_accesses', type=int, help='min number of accesses before we start'\
                        'predicting (session needs to build history)')
    parser.add_argument('--thresh_session', type=int, help='min number of accesses in session that'\
                        'need to be malicious before classifying the session as such', default=1)
    parser.add_argument('--outdir', help='output directory to store session''s tpr/fpr in')
    parser.add_argument('--feature_colnames_file', help='file containing the column names of features')
    parser.add_argument('--feature_types', type=parse_feature_types, default='SPC',\
                        help='Types of features to use. S: Self-reported. P: Past behavior. C: Contextual.')
    parser.add_argument('--min_time_to_mal', type=int, nargs='+', default=[0], help='minimum time before visit to malicious site.')
    parser.add_argument('--scaler_type', type=str, default='minmax',\
                        help='type of feature scaler used (standard/minmax/robust/quantile/quantile_gauss/normalizer)')
    return parser.parse_args()

"""
parse (and check) the feature_types argument
"""
def parse_feature_types(in_arg):
    in_arg = str(in_arg).upper()
    for c in in_arg:
        if not c in 'SPC':
            raise Exception('Feature type can only include the letters \'C\', \'P\', or \'S\'. ')
    return in_arg

"""
classify access
"""
def classify_access(model, scaler, feature_colnames, accesses):
    X = accesses[feature_colnames].values
    X = np.asarray(X, dtype=np.float)
    X = scaler.transform(X) # normalize the data
    predictions = model.predict(X)
    ret = accesses[['session_id', 'hash', 'communication_start_time','to_mal_sec']]
    ret.loc[:,'prob'] = pd.Series( predictions[:,1], index=ret.index)
    return ret

"""
write results (mainly tpr and fpr for sessions) to file
"""
def store_results(fout, evals):
    fout = open(fout, mode='w')
    fieldnames = ['tp', 'fp','tn','fn','tpr','fpr']
    writer = csv.DictWriter(fout, fieldnames, lineterminator='\n', delimiter='\t')
    writer.writeheader()
    for val in evals:
        writer.writerow(val)
    fout.close()

def output_auc(auc_roc, output_file):
    fout = open(output_file, mode='w')
    fout.write('auc_roc\n%0.08f'%auc_roc)
    fout.close()

def output_prob(df_probs, output_file):
    df_probs.to_csv( path_or_buf=output_file, sep='\t', index=False, header=False, line_terminator='\n',
           columns=['hash','session_id','communication_start_time','prob'], float_format='%.08f' )

# use to select test samples and label them
args = parse_args()
test_min_time = args.test_min_time
test_max_time = args.test_max_time
min_accesses = args.min_accesses
thresh_session = args.thresh_session
data_file = args.features_file
outdir = args.outdir
feature_colnames_file = args.feature_colnames_file
feature_types = args.feature_types
min_time_to_mal = args.min_time_to_mal

# load neural network and scaler
model = keras.models.load_model(args.model_path)
scaler = joblib.load(args.normalization_path)

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

# read test data, and classify ...
df_probs_benign = None
df_probs_malicious = None
reader = pd.read_table( data_file, chunksize=10000, dtype={"customer":"object", "hash":"object", "sessionid":"object"} )
for rows in reader:
    rows = rows[ ( rows['communication_start_time'] >= test_min_time ) & ( rows['communication_start_time'] <= test_max_time )  ]
    rows = rows[ rows['Contextual:session_len_access'] >= min_accesses  ]
    rows_benign = rows[ rows['mal'] == 0 ]
    rows_malicious = rows[ rows['mal'] != 0 ]
    rows_malicious = rows_malicious[ rows_malicious['from_mal_sec'] < 0 ]
    rows_malicious = rows_malicious[ rows_malicious['to_mal_sec'] >= 0 ]
    
    if rows_benign.shape[0] > 0:
        df = classify_access(model, scaler, feature_colnames, rows_benign)
        if df_probs_benign is None:
            df_probs_benign = df
        else:
            df_probs_benign = df_probs_benign.append(df)
    if rows_malicious.shape[0] > 0:
        df = classify_access(model, scaler, feature_colnames, rows_malicious)
        if df_probs_malicious is None:
            df_probs_malicious = df
        else:
            df_probs_malicious = df_probs_malicious.append(df)

# evaluate the neural network for different thresholds
df_sessions_benign = df_probs_benign.groupby('session_id')['prob'].count()
for to_mal in min_time_to_mal:
    # print summary stats
    df_probs_malicious_to_mal = df_probs_malicious[ df_probs_malicious['to_mal_sec'] >= to_mal ]
    df_sessions_malicious_to_mal = df_probs_malicious_to_mal.groupby('session_id')['prob'].count()
    
    n_accesses_benign = df_probs_benign.shape[0]
    n_accesses_malicious = df_probs_malicious_to_mal.shape[0]
    n_sessions_benign = df_sessions_benign.shape[0]
    n_sessions_malicious = df_sessions_malicious_to_mal.shape[0]
    print '# benign sessions: ', n_sessions_benign, ', # benign accesses: ', n_accesses_benign
    print '# malicious sessions: ', n_sessions_malicious, ', # malicious accesses: ', n_accesses_malicious
    
    evals_access = []
    evals_session = []

    step = 0.001
    for thresh in np.arange(0,1+step,step):

        df_a_b = df_probs_benign[ df_probs_benign['prob'] >= thresh ]
        df_a_m = df_probs_malicious_to_mal[ df_probs_malicious_to_mal['prob'] >= thresh ]
        tp_access = float(df_a_m.shape[0])
        fp_access = float(df_a_b.shape[0])
        tn_access = n_accesses_benign - fp_access
        fn_access = n_accesses_malicious - tp_access

        sr_s_b = df_a_b.groupby('session_id')['prob'].count()
        sr_s_b = sr_s_b[ sr_s_b >= thresh_session ]
        sr_s_m = df_a_m.groupby('session_id')['prob'].count()
        sr_s_m = sr_s_m[ sr_s_m >= thresh_session ]
        tp_session = float(sr_s_m.shape[0])
        fp_session = float(sr_s_b.shape[0])
        tn_session = n_sessions_benign - fp_session
        fn_session = n_sessions_malicious - tp_session

        # update tprs, fprs, ... for access-level predictions
        val = {}
        val['tp'] = tp_access
        val['fp'] = fp_access
        val['tn'] = tn_access
        val['fn'] = fn_access
        val['tpr'] = tp_access/n_accesses_malicious
        val['fpr'] = fp_access/n_accesses_benign
        evals_access.append(val)

        # update tprs, fprs, ... for session-level predictions
        val = {}
        val['tp'] = tp_session
        val['fp'] = fp_session
        val['tn'] = tn_session
        val['fn'] = fn_session
        val['tpr'] = tp_session/n_sessions_malicious
        val['fpr'] = fp_session/n_sessions_benign
        evals_session.append(val)

        # display result
        print 'thresh = %0.4f, tp_access = %0.4f, tp_session = %0.4f, fp_access = %0.4f,'\
              ' fp_session = %0.4f'\
              %( thresh, tp_access/n_accesses_malicious, tp_session/n_sessions_malicious,\
                 fp_access/n_accesses_benign, fp_session/n_sessions_benign )
        
    # auc
    auc_roc_session = auc( [ val['fpr'] for val in evals_session], [ val['tpr'] for val in evals_session])
    auc_roc_access = auc( [ val['fpr'] for val in evals_access], [ val['tpr'] for val in evals_access])

    # write results to file
    fname = __file__.split('/')[-1]
    fname = fname[:-3] + '-' + str(thresh_session) + '-' +\
            feature_types + '-' + args.scaler_type + '-' +\
            str(to_mal) + '-' +\
            str(test_min_time) + 'TO' + str(test_max_time)
    fname = os.path.join(outdir, fname)
    out_fname = fname + '.tsv'        
    out_access_fname = fname + '_access.tsv'
    session_auc_fname = fname + '_auc.tsv' 
    access_auc_fname = fname + '_access_auc.tsv' 
    benignid_fname = fname + '_benignid.tsv'
    maliciousid_fname = fname + '_maliciousid.tsv'
    store_results(out_fname, evals_session)
    store_results(out_access_fname, evals_access)
    output_auc(auc_roc_session, session_auc_fname)
    output_auc(auc_roc_access, access_auc_fname)
    output_prob(df_probs_benign, benignid_fname)
    output_prob(df_probs_malicious_to_mal, maliciousid_fname)
