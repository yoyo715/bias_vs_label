

import pprint, pickle

pkl_file = open('/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/label_output/test_trial0_epoch0', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(len(data1['Y_predicted'][0]))

pkl_file.close()