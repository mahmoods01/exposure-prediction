import csv
from datetime import datetime
from UserStudy import *
import user_agents


DAY_NBINS = 24            # number of bins to split a day into
N_TOP_DOMAINS = 1e5       # how many top sites to keep?
N_BROWSERS = 3            # Chrome, Safari, or other
N_OS = 3                  # Android, iOS, other
N_TOPICS = 99             # Num. of DigitalArts topics


"""
Maintain the state for a customer
"""
class Customer:

    def __init__(self, customer_code):
        global DAY_NBINS
        global N_TOPICS
        self.code = customer_code
        self.session_count = 0
        self.access_count = 0
        self.bytes_down = 0
        self.bytes_up = 0
        self.time_online_seconds = 0
        self.n_visits_to_non_top = 0
        self.n_visits_malicious = 0
        self.times_active = [0]*DAY_NBINS
        self.topics = [0]*N_TOPICS

    def update_state(self, session):
        self.session_count += 1
        self.access_count += session.access_count
        self.bytes_down += session.bytes_down
        self.bytes_up += session.bytes_up
        self.time_online_seconds += session.length_seconds
        self.n_visits_to_non_top += session.n_visits_to_non_top
        self.n_visits_malicious += session.visited_malicious
        self.times_active = [sum(x) for x in zip(self.times_active, session.times_active)]
        self.topics = [sum(x) for x in zip(self.topics, session.topics)]

"""
Maintain the state for a session
"""
class Session:

    def __init__(self, log_entry, malicious_domains, top_domains):
        global DAY_NBINS
        global N_TOPICS
        global N_OS
        global N_BROWSERS
        self.session_id = log_entry['session']
        self.customer_code = log_entry['custo']
        self.time_start = log_entry['communication_start_time']
        self.session_last_entry = log_entry['last_entry_id']
        start_datetime = datetime.strptime(str(self.time_start), '%Y%m%d%H%M%S')
        self.start_time_seconds = int((start_datetime-datetime(1970, 1, 1)).total_seconds())
        self.started_on_weekend = 1*(start_datetime.weekday()>=5)
        self.access_count = 0
        self.length_seconds = 0
        self.bytes_up = 0
        self.bytes_down = 0
        self.n_visits_to_non_top = 0
        self.visited_malicious = 0
        self.times_active = [0]*DAY_NBINS
        self.topics = [0]*N_TOPICS
        self.browser = [0]*N_BROWSERS
        self.os = [0]*N_OS
        self.update_state(log_entry, malicious_domains, top_domains)

    def update_state(self, log_entry, malicious_domains, top_domains):
        current_domain = log_entry['url_domain']
        # simple fields
        self.access_count += 1
        self.bytes_up += log_entry['send_byte_number']
        self.bytes_down += log_entry['receive_byte_number']
        # was a malicious domain visited?
        if self.visited_malicious==0:
            self.visited_malicious = 1*(log_entry['mal']=='malicious')
        # visits to non top domains
        self.n_visits_to_non_top += 1*(current_domain in top_domains)
        # session's length in seconds
        current_time_seconds =\
                        int((datetime.strptime(str(log_entry['communication_start_time']), '%Y%m%d%H%M%S')-datetime(1970, 1, 1)).total_seconds())
        self.length_seconds = current_time_seconds-self.start_time_seconds
        # update activity time
        hhmmss = log_entry['communication_start_time']%1000000
        seconds_in_day = (hhmmss/10000)*3600 + ((hhmmss%10000)/100)*60 + hhmmss%100
        bin_size = 86400 / DAY_NBINS # <- (24*60*60)/DAY_NBINS
        day_bin = seconds_in_day / bin_size
        self.times_active[day_bin] = 1
        # update topic distribution
        topics = [int(v) for v in log_entry['cat_cd_url'].split(',')]
        for t in topics:
            if t<0 or t>=N_TOPICS:
                self.topics[0] += 1./len(topics)
            else:
                self.topics[t] += 1./len(topics)
        # update os and browser
        u_a = user_agents.parse(log_entry['useragent'])
        if u_a.os.family=='iOS':
            self.os[0] = 1
        elif u_a.os.family=='Android':
            self.os[1] = 1
        else:
            self.os[2] = 1
        if u_a.browser.family=='Chrome':
            self.browser[0] = 1
        elif u_a.browser.family=='Mobile Safari UI/WKWebView':
            self.browser[1] = 1
        else:
            self.browser[2] = 1
    
"""
SelfReported answers by participants
"""
class SelfReported:

    feature_types = [\
                'gender',\
                'antivirus',\
                'app_market',\
                'proceed_on_warning',\
                'RSeBIS',\
                'confidence',\
                'past_compromise'\
    ]
    CODE_COL = 'CUSTCD'

    """
    Read questionnaire answers from file
    and create features
    """
    def __init__(self, questionnaire):
        self.customer_features = dict()
        q_file = open(questionnaire, mode='r')
        reader = csv.DictReader(q_file, delimiter='\t')
        for row in reader:
            customer_code = row[self.CODE_COL]
            x = UserStudy.compute_features(row, self.feature_types)
            self.customer_features[customer_code] = x

    """
    Get features--return values from the precomputed
    dictionary
    """
    def get_features(self, customer_code):
        if customer_code in self.customer_features:
            return self.customer_features[customer_code]
        return [0]*len(self.feature_types)

    """
    Textual description of SelfReported features
    """
    def feature_description(self):
        desc = UserStudy.feature_description(self.feature_types)
        desc = ['SelfReported:'+v for v in desc]
        return desc

"""
Past behavior of participants
"""
class PastBehavior:

    def __init__(self):
        self.customers = dict() # data strutcture to maintain customer's state

    """
    Update the customers data structure
    """
    def update_state(self, session):
        customer = self.customers[session.customer_code]
        customer.update_state(session)

    """
    Get PastBehavior features
    """
    def get_features(self, customer_code, n_days):
        x = []
        # get customer's state
        if customer_code in self.customers:
            # seen before
            customer = self.customers[customer_code]
        else:
            # first encounter
            customer = Customer(customer_code)
            self.customers[customer_code] = customer
        # avg. number of sessions and accesses per day
        x.append( float(customer.session_count)/n_days )
        x.append( float(customer.access_count)/n_days )
        # avg. number of bytes downloaded/uploaded per day
        x.append( float(customer.bytes_down)/n_days )
        x.append( float(customer.bytes_up)/n_days )
        # avg. session length in seconds and # accesses
        if customer.session_count==0:
            x += [0, 0]
        else:
            x.append( float(customer.time_online_seconds)/customer.session_count )
            x.append( float(customer.access_count)/customer.session_count )
        # fraction of visits to non-top sites in session
        if customer.access_count==0:
            x.append(0)
        else:
            x.append( float(customer.n_visits_to_non_top)/customer.access_count )
        # whether visited to malicious sites and fraction of sessions in which this happens
        if customer.session_count==0:
            x += [0, 0]
        else:
            x.append( 1*(customer.n_visits_malicious>0) )
            x.append( float(customer.n_visits_malicious)/customer.session_count )
        # browsing rate at different times (normalized by # accesses)
        if customer.session_count==0:
            x += customer.times_active
        else:
            x += [float(v)/customer.session_count for v in customer.times_active]
        # topics of sites visited (normalized by # accesses)
        if customer.session_count==0:
            x += customer.topics
        else:
            x += [float(v)/customer.session_count for v in customer.topics]
        return x

    """
    Description of PastBehavior features
    """
    def feature_description(self):
        global DAY_NBINS, N_TOPICS
        desc = [\
                'avg_day_session',\
                'avg_day_access',\
                'avg_byte_down',\
                'avg_byte_up',\
                'avg_session_len_sec',\
                'avg_session_len_access',\
                'frac_visits_non_top',\
                'visited_malicious',\
                'frac_malicious_visit'\
        ]
        desc += ['day_activity_'+str(i) for i in range(DAY_NBINS)]
        desc += ['topic_popularity_'+str(i) for i in range(N_TOPICS)]
        desc = ['PastBehavior:'+v for v in desc]
        return desc

"""
Contextual features that describe the
customer's current session
"""
class ContextualFeatures:

    def __init__(self, alexatop1m_path):
        global N_TOP_DOMAINS
        self.sessions = dict()         # data structure to maintain state of active sessions
        self.malicious_domains = set() # set of known malicious domains
        self.top_domains = set()       # top domains from alexa
        reader = open(alexatop1m_path)
        for row in reader:
            if len(self.top_domains)>=N_TOP_DOMAINS:
                break;
            self.top_domains.add(row[:-1].split(',')[1])

    """
    Update the state of the domains graph and
    the session structure, and returns any session
    that has ended
    """
    def update_state(self, log_entry):
        # update set of malicious domains
        domain = log_entry['url_domain']
        mal = log_entry['mal']
        if mal=='malicious':
            self.malicious_domains.add(domain)
        # update sessions
        session_id = log_entry['session']
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_state(log_entry, self.malicious_domains, self.top_domains)
        else:
            session = Session(log_entry, self.malicious_domains, self.top_domains)
            self.sessions[session_id] = session
        # check if session has ended, if so return it so it can be used
        # to update the state of the PastBehavior feature extractor
        if session.session_last_entry==log_entry['id']:
            del self.sessions[session_id]
            return session
        return None

    """
    Get ContextualFeatures
    """
    def get_features(self, log_entry):
        session_ended = self.update_state(log_entry)
        x = []
        session_id = log_entry['session']
        if session_ended==None:
            session = self.sessions[session_id]
        else:
            session = session_ended
        # session's length, in seconds and # accesses
        x.append( session.length_seconds )
        x.append( session.access_count )
        # number of bytes downloaded and uploaded
        x.append( session.bytes_down )
        x.append( session.bytes_up )
        # fraction of visits to non-top domains
        x.append( float(session.n_visits_to_non_top)/session.access_count )
        # is it weekend?
        x.append( session.started_on_weekend )
        # time of day
        x += session.times_active
        # fraction of each topic for every visit
        x += [float(v)/session.access_count for v in session.topics]
        # Os and Browser info
        x += session.os
        x += session.browser
        return x, session_ended

    """
    Description of ContextualFeatures
    """
    def feature_description(self):
        global DAY_NBINS, N_TOPICS, N_OS, N_BROWSERS
        desc = [\
                'session_len_sec',\
                'session_len_access',\
                'session_bytes_down',\
                'session_bytes_up',\
                'session_fraction_non_top',\
                'weekend'\
        ]
        desc += ['time_active_'+str(i) for i in range(DAY_NBINS)]
        desc += ['session_topics_'+str(i) for i in range(N_TOPICS)]
        desc += ['os_'+str(i) for i in range(N_OS)]
        desc += ['browser_'+str(i) for i in range(N_BROWSERS)]
        desc = ['Contextual:'+v for v in desc]
        return desc
