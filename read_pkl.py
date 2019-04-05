import pickle
from sklearn.metrics import confusion_matrix
import numpy as np

fname = 'NEW_wfasttext_RUN0_EPOCH0_.pkl'


pkl_fileft = open(fname, 'rb')
dataft = pickle.load(pkl_fileft)

betas = dataft['betas']
print(betas.shape)

print(dataft['Y_STRAIN'].shape)
print(dataft['Y_SVAL'].shape)
print(dataft['Y_RTEST'].shape)
print(dataft['Y_RVAL'].shape)
print(dataft['Y_STEST'].shape)

print(dataft['yhat_strain'].shape)
print(dataft['yhat_sval'].shape)
print(dataft['yhat_rtest'].shape)
print(dataft['yhat_rval'].shape)
print(dataft['yhat_stest'].shape)


prediction_max = np.argmax(dataft['yhat_strain'], axis=0)
true_label_max = np.argmax(dataft['Y_STRAIN'], axis=1)

class_error = np.sum(true_label_max != prediction_max.T) * 1.0 / dataft['Y_STRAIN'].shape[0]

print(confusion_matrix(true_label_max, prediction_max))
print(class_error)

pkl_fileft.close()
