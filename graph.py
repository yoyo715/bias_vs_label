# graph.py


from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pandas as pd
import numpy as np

def remove_brackets(df):
    i = 0
    while i < len(df.columns):
        df[i] = df[i].str.strip("[]").astype('float64') 
        i += 1
    return df


def graph_loss(losses_train, losses_test, losses_manual, EPOCH):    
    epochs = [l for l in range(EPOCH)]
    
    # train
    losses_train = losses_train.drop(losses_train.columns[-1],axis=1)
    losses_train = remove_brackets(losses_train)
    summary_train = losses_train.describe()
    mean_train = np.array(summary_train.loc[['mean']])
    std_train = np.array(summary_train.loc[['std']])
    print(std_train)
    mean_train.resize((EPOCH))
    std_train.resize((EPOCH))
    
    # test
    losses_test = losses_test.drop(losses_test.columns[-1],axis=1)
    losses_test = remove_brackets(losses_test)
    summary_test = losses_test.describe()
    mean_test = np.array(summary_test.loc[['mean']])
    std_test = np.array(summary_test.loc[['std']])
    mean_test.resize((EPOCH))
    std_test.resize((EPOCH))
    
    # manual
    losses_manual = losses_manual.drop(losses_manual.columns[-1],axis=1)
    losses_manual = remove_brackets(losses_manual)
    summary_manual = losses_manual.describe()
    mean_manual = np.array(summary_manual.loc[['mean']])
    std_manual = np.array(summary_manual.loc[['std']])
    mean_manual.resize((EPOCH))
    std_manual.resize((EPOCH))
    
    
    plt.errorbar(epochs, mean_train, yerr=std_train, fmt='-o', marker='s',  mfc='orange', barsabove=True, capsize=5, label="training loss")
    plt.errorbar(epochs, mean_test, yerr=std_test, fmt='-o', marker='s', mfc='orange', barsabove=True, capsize=5, label="testing loss")
    plt.errorbar(epochs, mean_manual, yerr=std_manual, fmt='-o', marker='s', mfc='orange', barsabove=True, capsize=5, label="manual loss")
    
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    
def graph_error(class_error_train, class_error_test, class_error_manual, EPOCH): 
    epochs = [l for l in range(EPOCH)]
    
    # train
    class_error_train = class_error_train.drop(5, axis=1)
    summary_train = class_error_train.describe()
    print(class_error_train)
    print(summary_train)
    mean_train = np.array(summary_train.loc[['mean']])
    std_train = np.array(summary_train.loc[['std']])
    mean_train.resize((EPOCH))
    std_train.resize((EPOCH))
    
    # test
    #class_error_test = class_error_test.drop(5, axis=1)
    summary_test = class_error_test.describe()
    mean_test = np.array(summary_test.loc[['mean']])
    std_test = np.array(summary_test.loc[['std']])
    mean_test.resize((EPOCH))
    std_test.resize((EPOCH))
    
    # manual
    #class_error_manual = class_error_manual.drop(5, axis=1)
    summary_manual = class_error_manual.describe()
    mean_manual = np.array(summary_manual.loc[['mean']])
    std_manual = np.array(summary_manual.loc[['std']])
    mean_manual.resize((EPOCH))
    std_manual.resize((EPOCH))
    
    plt.errorbar(epochs, mean_train, yerr=std_train, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="training classification error")
    plt.errorbar(epochs, mean_test, yerr=std_test, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="testing classification error")
    plt.errorbar(epochs, mean_manual, yerr=std_manual, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="manual classification error")
    
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    
def graph_auc(AUC_train, AUC_test, AUC_manual, EPOCH): 
    epochs = [l for l in range(EPOCH)]
    
    # train
    summary_train = AUC_train.describe()
    mean_train = np.array(summary_train.loc[['mean']])
    std_train = np.array(summary_train.loc[['std']])
    mean_train.resize((EPOCH))
    std_train.resize((EPOCH))
    
    # test
    summary_test = AUC_test.describe()
    mean_test = np.array(summary_test.loc[['mean']])
    std_test = np.array(summary_test.loc[['std']])
    mean_test.resize((EPOCH))
    std_test.resize((EPOCH))
    
    # manual
    summary_manual = AUC_manual.describe()
    mean_manual = np.array(summary_manual.loc[['mean']])
    std_manual = np.array(summary_manual.loc[['std']])
    mean_manual.resize((EPOCH))
    std_manual.resize((EPOCH))
    
    plt.errorbar(epochs, mean_train, yerr=std_train, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="training AUC scores")
    plt.errorbar(epochs, mean_test, yerr=std_test, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="testing AUC scores")
    plt.errorbar(epochs, mean_manual, yerr=std_manual, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="manual AUC scores")
    
    plt.ylabel('AUC scores')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
        
        
def graph_f1(F1_train, F1_test, F1_manual, EPOCH): 
    epochs = [l for l in range(EPOCH)]
    
    # train
    summary_train = F1_train.describe()
    mean_train = np.array(summary_train.loc[['mean']])
    std_train = np.array(summary_train.loc[['std']])
    mean_train.resize((EPOCH))
    std_train.resize((EPOCH))
    
    # test
    summary_test = F1_test.describe()
    mean_test = np.array(summary_test.loc[['mean']])
    std_test = np.array(summary_test.loc[['std']])
    mean_test.resize((EPOCH))
    std_test.resize((EPOCH))
    
    # manual
    summary_manual = F1_manual.describe()
    mean_manual = np.array(summary_manual.loc[['mean']])
    std_manual = np.array(summary_manual.loc[['std']])
    mean_manual.resize((EPOCH))
    std_manual.resize((EPOCH))
    
    plt.errorbar(epochs, mean_train, yerr=std_train, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="training F1")
    plt.errorbar(epochs, mean_test, yerr=std_test, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="testing F1")
    plt.errorbar(epochs, mean_manual, yerr=std_manual, fmt='-o', marker='s',
                 mfc='orange', barsabove=True, capsize=5, label="manual F1")
    
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    
def graph_roc(train_FPR, train_TPR, test_FPR, test_TPR, manual_FPR, manual_TPR, train_AUC, test_AUC, manual_AUC):
    epochs = [l for l in range(EPOCH)]
    
    # train
    summary_trainFPR = train_FPR.describe()
    mean_trainFPR = np.array(summary_trainFPR.loc[['mean']])
    mean_trainFPR.resize((EPOCH))
    
    summary_trainTPR = train_TPR.describe()
    mean_trainTPR = np.array(summary_trainTPR.loc[['mean']])
    mean_trainTPR.resize((EPOCH))
    
    summary_trainAUC = train_AUC.describe()
    mean_trainAUC = np.array(summary_trainAUC.loc[['mean']])
    mean_trainAUC.resize((EPOCH))
    
    # test
    summary_testFPR = test_FPR.describe()
    mean_testFPR = np.array(summary_testFPR.loc[['mean']])
    mean_testFPR.resize((EPOCH))
    
    summary_testTPR = test_TPR.describe()
    mean_testTPR = np.array(summary_testTPR.loc[['mean']])
    mean_testTPR.resize((EPOCH))
    
    summary_testAUC = test_AUC.describe()
    mean_testAUC = np.array(summary_testAUC.loc[['mean']])
    mean_testAUC.resize((EPOCH))
    
    # manual
    summary_manualFPR = manual_FPR.describe()
    mean_manualFPR = np.array(summary_manualFPR.loc[['mean']])
    mean_manualFPR.resize((EPOCH))
    
    summary_manualTPR = manual_TPR.describe()
    mean_manualTPR = np.array(summary_manualTPR.loc[['mean']])
    mean_manualTPR.resize((EPOCH))
    
    summary_manualAUC = manual_AUC.describe()
    mean_manualAUC = np.array(summary_manualAUC.loc[['mean']])
    mean_manualAUC.resize((EPOCH))
    
    plt.title('FINAL Receiver Operating Characteristic (ROC curve)')
    plt.plot([0,1],[0,1],'r--')
    plt.plot(mean_trainFPR, mean_trainTPR, 'm', label="mean training AUC score = %f" % mean_trainAUC)
    plt.plot(mean_testFPR, mean_testTPR, 'c', label="mean testing AUC score = %f" % mean_testAUC)
    plt.plot(mean_manualFPR, mean_manualTPR, 'c', label="mean manual AUC score = %f" % mean_manualAUC)
    plt.ylabel('mean TPR')
    plt.xlabel('mean FPR')
    plt.legend(loc='upper left')
    plt.show()


def graph_precrecall(recall_train, recall_test, recall_manual, prec_train, prec_test, prec_manual):  
    # train
    summary_trainrecall = recall_train.describe()
    mean_trainrecall = np.array(summary_trainrecall.loc[['mean']])
    mean_trainrecall.resize((EPOCH))
    
    summary_trainprec = prec_train.describe()
    mean_trainprec = np.array(summary_trainprec.loc[['mean']])
    mean_trainprec.resize((EPOCH))
    
    # test
    summary_testrecall = recall_test.describe()
    mean_testrecall = np.array(summary_testrecall.loc[['mean']])
    mean_testrecall.resize((EPOCH))
    
    summary_testprec = prec_test.describe()
    mean_testprec = np.array(summary_testprec.loc[['mean']])
    mean_testprec.resize((EPOCH))
    
    # manual
    summary_manualrecall = recall_manual.describe()
    mean_manualrecall = np.array(summary_manualrecall.loc[['mean']])
    mean_manualrecall.resize((EPOCH))
    
    summary_manualprec = prec_manual.describe()
    mean_manualprec = np.array(summary_manualprec.loc[['mean']])
    mean_manualprec.resize((EPOCH))
    
    
    plt.plot(mean_trainrecall, mean_trainprec, 'm', label="mean training")
    plt.plot(mean_testrecall, mean_testprec, 'c', label="mean testing")
    plt.plot(mean_manualrecall, mean_manualprec, 'g', label="mean manual")
    plt.ylabel('Mean Precision')
    plt.xlabel('Mean Recall')
    plt.legend(loc='upper left')
    plt.show()
            

def main():    
    EPOCH = 20   # WARNING: must match main.py
        
    df = pd.read_csv('test.txt', sep=" ", header=None)
    
    loss_train = pd.read_csv('output/loss_train.txt', sep=",", header=None)
    loss_test = pd.read_csv('output/loss_test.txt', sep=",", header=None)
    loss_manual = pd.read_csv('output/loss_manual.txt', sep=",", header=None)

    error_train = pd.read_csv('output/error_train.txt', sep=",", header=None)
    error_test = pd.read_csv('output/error_test.txt', sep=",", header=None)
    error_manual = pd.read_csv('output/error_manual.txt', sep=",", header=None)
                
    precision_train = pd.read_csv('output/precision_train.txt', sep=",", header=None)
    precision_test = pd.read_csv('output/precision_test.txt', sep=",", header=None)
    precision_manual = pd.read_csv('output/precision_manual.txt', sep=",", header=None)
        
    recall_train = pd.read_csv('output/recall_train.txt', sep=",", header=None)
    recall_test = pd.read_csv('output/recall_test.txt', sep=",", header=None)
    recall_manual = pd.read_csv('output/recall_manual.txt', sep=",", header=None)
    
    F1_train = pd.read_csv('output/F1_train.txt', sep=",", header=None)
    F1_test = pd.read_csv('output/F1_test.txt', sep=",", header=None)
    F1_manual = pd.read_csv('output/F1_manual.txt', sep=",", header=None)
        
    AUC_train = pd.read_csv('output/AUC_train.txt', sep=",", header=None)
    AUC_test = pd.read_csv('output/AUC_test.txt', sep=",", header=None)
    AUC_manual = pd.read_csv('output/AUC_manual.txt', sep=",", header=None)


    graph_loss(loss_train, loss_test, loss_manual, EPOCH)
    #graph_error(error_train, error_test, error_manual, EPOCH)
    #graph_auc(AUC_train, AUC_test, AUC_manual, EPOCH)
    #graph_f1(F1_train, F1_test, F1_manual, EPOCH)
    #graph_roc(train_FPR, train_TPR, test_FPR, test_TPR, manual_FPR, manual_TPR, train_AUC, test_AUC, manual_AUC)
    #graph_precrecall(recall_train, recall_test, recall_manual, prec_train, prec_test, prec_manual)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
