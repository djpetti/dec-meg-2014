"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

import math
import numpy as np
import pylab as pl
import sys, getopt
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, RandomizedLogisticRegression, lasso_stability_path, LassoLarsCV
from sklearn import cross_validation, metrics
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV,RFE,SelectPercentile, f_classif
from sklearn.grid_search import GridSearchCV
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype='bandstop')
    return(b,a)


def butter_bandpass_filter(data,lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a,data)
    return y

def filter_noise(XX,t1,t2,sfreq):
    #t = np.linspace(t1,t2,1.5/375,endpoint=False)
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    #print beg, end

    b, a = butter_bandpass(49, 51, sfreq, order=5)
    b1, a1 = butter_bandpass(99, 101, sfreq, order=5)
    

    f0 = 50
    XXnew = XX
    count =0
    for dimX in XXnew:
        for dimY in dimX:
            y = dimY[beg:end]
            #ynew = butter_bandpass_filter(y, 45, 55, sfreq, order=5)
            ynew = lfilter(b, a, y)
            ynew2 = lfilter(b1, a1, ynew)
            dimY[beg:end] = ynew2
            #print count
	    count += 1 
    return XXnew         

def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    #print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    #print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    #print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':


    print "Parsing arguments"
    plotfile = ''
    algo = ''
    reg = 1
    mode = 0
    fe=-1
    print sys.argv[1:]
    resfile = ''  
 
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hm:a:p:r:c:f:",["mode","algo=","pfile=","reg=","fe="])
    except getopt.GetoptError:
      print 'ensemble.py -o <plotfile>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'ensemble.py -o <plotfile> -m  mode(1=modelselection 0=specific model) -C <regularization> -a <algorithm(l1|l2|lsvm|ksvm) -f <feature extraction 0=None 1=l1 based 2=chi2>'
         sys.exit()
      elif opt in ("-m", "--mode"):  #mode= 0 clf.model 1 model selection
         mode = float(arg)
      elif opt in ("-a", "--algo"):
         algo = arg
      elif opt in ("-p", "--pfile"):
         plotfile = arg
      elif opt in ("-r", "--rfile"):
         resfile = arg
      elif opt in ("-c", "--reg"):
         reg = float(arg)
      elif opt in ("-f", "--fe"):
         fe = int(arg)
    print 'Algorithm is ', algo
    print 'Plot file is ', plotfile
    print 'Regularization Parameter is ', reg
    

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    subjects_train = [1, 7, 8, 13, 15, 5, 3, 10, 2, 6, 12, 14, 4, 11, 9, 16] # use range(1, 17) for all subjects
    #subjects_train = [1] # use range(1, 17) for all subjects
    #print "Training on subjects", subjects_train 
    subjects_val = []

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0
    tmax = 0.5
    print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    X_val = []
    y_val = []

    print
    #print "Creating the trainset."
    for subject in subjects_train:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        #print "Dataset summary:"
        #print "XX:", XX.shape
        #print "yy:", yy.shape
        #print "sfreq:", sfreq

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

	#plot_freq_y(XX, yy, -0.5, 1.0, sfreq, 1)

        XX = create_features(XX, tmin, tmax, sfreq)
        #pl.show()

        #print "XX:", XX.shape
        X_train.append(XX)
        y_train.append(yy)

    for subject in subjects_val:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading val", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_val.append(XX)
        y_val.append(yy)


    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    #X_val = np.vstack(X_val)
    #y_val = np.concatenate(y_val)

    n_features = X_train.shape[1]   
    print "XX:", n_features

    g =   1.0/float((3*n_features))
    print g
 
    clf_lsvm = svm.SVC(kernel='linear', C=10)
    clf_ksvm = svm.SVC(kernel='rbf', C=10, gamma=g )
    clf_lasso = LogisticRegression(C=1000,penalty='l1',random_state=0)
    clf_ridge = LogisticRegression(C=0.01,penalty='l2',random_state=0) 
    clf_rf = RandomForestClassifier(n_estimators=850, max_depth=None, max_features=int(math.sqrt(n_features)), min_samples_split=100, random_state=144, n_jobs=4);
    clf_etree = ExtraTreesClassifier(n_estimators=1000, max_depth=None, max_features=int(math.sqrt(n_features)), min_samples_split=100, random_state=144, n_jobs=4);
    clf_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=500, random_state=74494, learning_rate=0.8) 
    clf_gboost = GradientBoostingClassifier(n_estimators=200, random_state=74494, learning_rate=0.2) 

    print "Training."
    clf_lsvm.fit(X_train, y_train)
    clf_ksvm.fit(X_train, y_train)
    clf_lasso.fit(X_train, y_train)
    clf_ridge.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)
    clf_etree.fit(X_train, y_train)
    clf_boost.fit(X_train, y_train)
    clf_gboost.fit(X_train, y_train)

    if(resfile != ''):
        print "Creating the testset."
        subjects_test = range(17, 24)
        for subject in subjects_test:
            filename = 'data/test_subject%02d.mat' % subject
            print "Loading", filename
            data = loadmat(filename, squeeze_me=True)
            XX = data['X']
            ids = data['Id']
            sfreq = data['sfreq']
            tmin_original = data['tmin']
            print "Dataset summary:"
            print "XX:", XX.shape
            print "ids:", ids.shape
            print "sfreq:", sfreq
            XX = filter_noise(XX, -0.5, 1.0, sfreq)
            XX = create_features(XX, tmin, tmax, sfreq)
            X_test.append(XX)
            ids_test.append(ids)

        X_test = np.vstack(X_test)
        ids_test = np.concatenate(ids_test)
        print "Testset:", X_test.shape
                      
        print "Predicting."
        y_lsvm = clf_lsvm.predict(X_test)
        y_ksvm = clf_ksvm.predict(X_test)
        y_lasso = clf_lasso.predict_proba(X_test)
        y_ridge = clf_ridge.predict_proba(X_test)
        y_rf = clf_rf.predict_proba(X_test)
        y_etree = clf_etree.predict_proba(X_test)
        y_boost = clf_boost.predict_proba(X_test)
        y_gboost = clf_gboost.predict_proba(X_test)

        filename_submission = "submission.csv"
        print "Creating submission file", resfile
        f = open(resfile, "w")
        print >> f, "Id,Prediction"
        for i in range(len(y_pred)):
            print >> f, str(ids_test[i]) + "," + str(y_lsvm[i]) + "," + str(y_ksvm[i]) + "," + str(y_lasso[i,1]) + "," + str(y_ridge[i,1]) + "," + str(y_rf[i,1]) + "," + str(y_etree[i,1]) + "," + str(y_boost[i,1]) + "," + str(y_gboost[i,1]) 
        f.close()

    print "Done."
    
