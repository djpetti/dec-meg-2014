"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

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
    #pl.subplot(3,1,1)
    #pl.plot(t,ynew, label='Filtered signal (%g Hz)' % f0)
    #pl.xlabel('time (seconds)')
    #n = len(ynew)
    #Ff = np.fft.rfft(ynew)
    #k = np.arange(n)
    #frq = k/1.5
    #frq = frq[range(n/2+1)]
    #pl.subplot(3,1,2)
    #print np.abs(Ff).shape + frq.shape
    #pl.plot(frq,np.abs(Ff),'r')
    #pl.xlabel('freq (Hz)')
    #pl.subplot(3,1,3)
    #pl.plot(frq,np.angle(Ff),'r')
    #pl.xlabel('freq (Hz)')

def plot_freq_y(XX, yy, t1, t2, sfreq, pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    #print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag0 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle0 = np.zeros(int((n/2)+1),dtype=float)
    Ffmag1 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle1 = np.zeros(int((n/2)+1),dtype=float)
    count = 0

    XX0 = XX[np.where(yy==0)]
    XX1 = XX[np.where(yy==1)]

    for dimX in XX0:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag0 += np.abs(f)
            Ffangle0 += np.angle(f)
            count +=1
           
    Ffmag0 = Ffmag0/count
    Ffangle0 = Ffangle0/count
   
    count = 0
    for dimX in XX1:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag1 += np.abs(f)
            Ffangle1 += np.angle(f)
            count +=1
           
    Ffmag1 = Ffmag1/count
    Ffangle1 = Ffangle1/count


    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]

    pl.subplot(2,1,1)
    pl.plot(frq,Ffmag0,'r')
    pl.plot(frq,Ffmag1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle0,'r')
    pl.plot(frq,Ffangle1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')




 
def plot_freq(XX,t1,t2,sfreq,pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag = np.zeros(int((n/2)+1),dtype=float)
    Ffangle = np.zeros(int((n/2)+1),dtype=float)
    count = 0
    for dimX in XX:
        for dimY in dimX:
            y = dimY[beg:end]
            #n = len(y)
            #print y.shape + t.shape
            f = np.fft.rfft(y)
            Ffmag += np.abs(f)
            Ffangle += np.angle(f)
            count +=1
 
    Ffmag = Ffmag/count
    Ffangle = Ffangle/count
    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]
    pl.subplot(2,1,1)
    #print np.abs(Ff).shape + frq.shape
    pl.plot(frq,Ffmag,'r')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')

    #pl.subplot(3,1,1)
    #pl.plot(t,y)
    #pl.xlabel('time (seconds)')



    return 0;

def cross_validate(clf, tr, te):
    scores = cross_validation.cross_val_score(clf, tr, te, cv=5 ) 
    print ("Cross Validation scores: mean=%f dev=%f" % (scores.mean(),scores.std()) )


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
      print 'dhruv.py -o <plotfile>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'dhruv.py -o <plotfile> -m  mode(1=modelselection 0=specific model) -C <regularization> -a <algorithm(l1|l2|lsvm|ksvm) -f <feature extraction 0=None 1=l1 based 2=chi2>'
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
    subjects_train = [1] # use range(1, 17) for all subjects
    #print "Training on subjects", subjects_train 
    subjects_val = [4, 11]

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0
    tmax = mode
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
    
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    n_features = X_train.shape[1]   
    print "XX:", n_features

    g =   1.0/float((3*n_features))
    print g


 
    clf = RandomForestClassifier(n_estimators=int(reg), max_depth=None,min_samples_split=1, random_state=144, n_jobs=4);
  
    print "Classifier:"
    print clf, clf.get_params()


    print "Training."
    clf.fit(X_train, y_train)
        
    print "Validation set score: " , clf.score(X_val, y_val)

    if(fe==1): #L1 norm based feature elimination
        clf_fe = LogisticRegression(C=1000,penalty='l1',random_state=0)
        clf_fe.fit(X_train, y_train)
        X_train = X_train[:,clf_fe.coef_.ravel()!=0]
        print "Xtrain.shape: ", X_train.shape
        X_val = X_val[:,clf_fe.coef_.ravel()!=0]

        clf2_l = svm.SVC(kernel='linear', C=reg)
        clf2_l.fit(X_train, y_train)
        print "Lasso Validation set score filtered coeff linear: " , clf2_l.score(X_val, y_val)
        clf2 = svm.SVC(kernel='rbf', C=reg, gamma=g)
        clf2.fit(X_train, y_train)
        print "Lasso Validation set score filtered coeff: " , clf2.score(X_val, y_val)
    elif(fe==4): #Univariate feature selection F-score based
        selector = SelectPercentile(f_classif, percentile=75)
        selector.fit(X_train, y_train)
        scores = -np.log10(selector.pvalues_)
	scores /= scores.max()
        #pl.figure(2)
        #pl.xlabel("Features")
        #pl.ylabel("F-scores")
	#pl.bar(scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g')
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)
        print "Xtrain.shape: ", X_train.shape
        clf2_l = svm.SVC(kernel='linear', C=reg)
	clf2 = svm.SVC(kernel='rbf', C=reg, gamma=g)
        clf2_l.fit(X_train, y_train)
        clf2.fit(X_train, y_train)
        print "Fscore Validation set score filtered coeff: " , clf2.score(X_val, y_val)
        print "Fscore Validation set score filtered coeff linear: " , clf2_l.score(X_val, y_val)

    pl.show()

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
        y_pred = clf.predict(X_test)

        filename_submission = "submission.csv"
        print "Creating submission file", resfile
        f = open(resfile, "w")
        print >> f, "Id,Prediction"
        for i in range(len(y_pred)):
            print >> f, str(ids_test[i]) + "," + str(y_pred[i])
        f.close()
    print "Done."
    
