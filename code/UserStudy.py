"""
Processing UserStudy data
"""

class UserStudy:
    """
    Use this to extract variables from responses
    to the UserStudy
    """

    feature_ids = {\
                   'age' : 'F2_AGE',\
                   'gender' : 'F1_SEX',\
                   'antivirus' : 'Q4',\
                   'app_market' : 'Q5S3',\
                   'proceed_on_warning' : 'Q6_',\
                   'RSeBIS' : 'Q7S',\
                   'confidence' : 'Q7S',\
                   'past_compromise' : 'Q3S'\
                   }

    @classmethod
    def compute_features(cls, row, features):
        """
        Given a list of feature names, and a row of
        the questionnaire data read by csv.DictReader
        (i.e., represented as a dictionary), compute
        the features ...
        """
        x = []
        for f in features:
            if f=='age':
                age = int(row[cls.feature_ids[f]])
                x.append( age )
            elif f=='gender':
                gen = int(row[cls.feature_ids[f]])
                gen = gen-1
                x.append( gen )
            elif f=='antivirus':
                av = int(row[cls.feature_ids[f]])
                if av==1:
                    x.append(1)
                else:
                    x.append(0)
            elif f=='app_market':
                app_market = int(row[cls.feature_ids[f]])
                if app_market>=2:
                    x.append(1)
                else:
                    x.append(0)
            elif f=='proceed_on_warning':
                on_warning = 0
                for i in range(1,4):
                    if int(row[cls.feature_ids[f]+str(i)])!=0:
                        on_warning = 1
                        break;
                x.append(on_warning)
            elif f=='RSeBIS':
                rsebis = 0
                for i in range(1,6):
                    rsebis += int(row[cls.feature_ids[f]+str(i)])
                x.append((rsebis-5)/(25.-5))
            elif f=='confidence':
                conf = 0
                for i in range(6,12):
                    conf += int(row[cls.feature_ids[f]+str(i)])
                x.append((conf-6)/(30.-6))
            elif f=='past_compromise':
                past_comp = 0
                for i in [1, 3, 4]:
                    if int(row[cls.feature_ids[f]+str(i)])>=3:
                        past_comp = 1
                        break;
                x.append(past_comp)
            else:
                raise ValueError('Unknown feature "%s"'%(f,))
        return x


    @classmethod
    def feature_description(cls, features):
        """
        get description of features
        """
        desc = []
        for f in features:
            if f=='age':
                desc.append('Age (in years)')
            elif f=='gender':
                desc.append('Is female?')
            elif f=='antivirus':
                desc.append('Uses antivirus')
            elif f=='app_market':
                desc.append('Use unofficial app market')
            elif f=='proceed_on_warning':
                desc.append('Proceeds on browser warning')
            elif f=='RSeBIS':
                desc.append('RSeBIS proactive awareness score')
            elif f=='confidence':
                desc.append('Self-confidence in security knowledge')
            elif f=='past_compromise':
                desc.append('Suffered from compromise?')
            else:
                raise ValueError('Unknown feature "%s"'%(f,))
        return desc
