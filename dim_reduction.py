# dim_reduction.py

# This script's goal is to plot the data in 2D space in order to visualize the distribution


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


from dictionary_updated import Dictionary2

import pandas as pd
import numpy as np


# args from Simple Queries paper
DIM=30
LR=0.0001
WORDGRAMS=3
MINCOUNT=2
MINN=3
MAXN=3
BUCKET=1000000

print("starting dictionary creation") 
    
# initialize training
dictionary = Dictionary2(WORDGRAMS, MINCOUNT, BUCKET)
nwords = dictionary.get_nwords()
nclasses = dictionary.get_nclasses()
    
#initialize testing
X_train, X_test, y_train, y_test = dictionary.get_train_and_test()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
N = dictionary.get_n_train_instances()
N_test = dictionary.get_n_test_instances()
    
print("Number of Train instances: ", N, " Number of Test instances: ", N_test)
ntrain_eachclass = dictionary.get_nlabels_eachclass_train()
ntest_eachclass = dictionary.get_nlabels_eachclass_test()
print("N each class TRAIN: ", ntrain_eachclass, " N each class TEST: ", ntest_eachclass)


raw_train_labels = dictionary.get_raw_train_labels()

pca = PCA(n_components=3).fit(X_train.toarray())
data2D = pca.transform(X_train.toarray())

principalDf = pd.DataFrame(data = data2D
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

df = pd.DataFrame(raw_train_labels, columns=['target'])
finalDf = pd.concat([principalDf, df], axis = 1)

colors = ['turquoise', 'darkorange']
targets = [0.0, 1.0]

fig = plt.figure()

ax = fig.add_subplot(1,1,1, projection='3d') 
ax.set_title('2 component PCA', fontsize = 15)


for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color)
ax.legend(targets)

plt.show()              





