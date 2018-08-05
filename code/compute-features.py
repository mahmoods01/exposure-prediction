"""
Compute features from HTTP log entries
"""

import argparse
import csv
import sys

import psycopg2
import psycopg2.extras

from FeatureExtraction import *

def parse_args():
    parser = argparse.ArgumentParser();
    parser.add_argument('--dbname', help='name of psql DB')
    parser.add_argument('--questionnaire', help='path to questionnaire responses')
    parser.add_argument('--outfile', help='path to save features at in csv format')
    parser.add_argument('--alexa1m', help='path to alexa top 1m sites')
    return parser.parse_args()

"""
Compute the features we use for predictions:
1- Self reported features
2- Past Behavior--from what we know about the customer
3- Contextual--from what we know about the current session
and connections between domains
"""
def compute_features(log_entry, n_days):
    customer_code = log_entry['custo']
    # self reported
    f1 = SelfReportedFE.get_features(customer_code)
    # past behavior
    f2 = PastBehaviorFE.get_features(customer_code, n_days)
    # contextual features
    f3, session_ended = ContextualFE.get_features(log_entry)
    return f1+f2+f3, session_ended

"""
Update state of feature extractors, and also other
counters, like day counter
"""
def update_state(log_entry, session_ended):
    global n_days, prev_day
    # update day counter
    curr_day = log_entry['communication_start_time']/1000000
    if curr_day>prev_day:
        n_days += 1
        prev_day = curr_day
    # update state of PastBehaviorFE, in case a session has ended
    if session_ended!=None:
        PastBehaviorFE.update_state(session_ended)

# parse arges
args = parse_args()

# init feature extractors
SelfReportedFE = SelfReported(args.questionnaire)
PastBehaviorFE = PastBehavior()
ContextualFE = ContextualFeatures(args.alexa1m)

# open outfile for writing, and write header
outfile = open( args.outfile, mode='w' )
writer = csv.writer(outfile)
desc = SelfReportedFE.feature_description()
desc += PastBehaviorFE.feature_description()
desc += ContextualFE.feature_description()
header = ['customer', 'entry_id', 'session_id', 'communication_start_time'] + desc
writer.writerow(header)

# Fetch entries, sorted by access time
conn = psycopg2.connect(dbname=args.dbname)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur.execute('SELECT LogEntries.id, LogEntries.session, LogEntries.CUSTO, '\
            'LogEntries.COMMUNICATION_START_TIME, LogEntries.URL_DOMAIN, '\
            'LogEntries.REF_DOMAIN, LogEntries.MAL, LogEntries.SEND_BYTE_NUMBER, '\
            'LogEntries.RECEIVE_BYTE_NUMBER, Sessions.last_entry_id, '\
            'LogEntries.CAT_CD_URL, LogEntries.USERAGENT '\
            'FROM LogEntries JOIN Sessions on LogEntries.session=Sessions.id '\
            'ORDER BY COMMUNICATION_START_TIME, COMMUNICATION_START_MILLISECONDS')

# compute features
n_days = 0
for log_entry in cur:
    if n_days==0:
        n_days = 1
        prev_day = log_entry['communication_start_time']/1000000
    features, session_ended = compute_features(log_entry, n_days)
    writer.writerow([log_entry['custo'], log_entry['id'], log_entry['session'], log_entry['communication_start_time'] ] + features)
    update_state(log_entry, session_ended)

# done writing
outfile.close()
