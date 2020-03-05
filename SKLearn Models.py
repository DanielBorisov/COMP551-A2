# -*- coding: utf-8 -*-
"""

COMP 551 - Assignment 2
Daniel Borisov
Fall 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

# DATA LOADING AND PROCESSING

rawX = sp.load('tokens/X_stem_sparse.pickle', allow_pickle = True)

dataY = sp.load('tokens/y_stem_sparse.pickle', allow_pickle = True)

tform = TfidfTransformer()
dataX = tform.fit_transform(rawX)


# LOGISTIC REGRESSION

logReg = LogisticRegression(solver='newton-cg',  max_iter=1000, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)
logReg2 = LogisticRegression(solver='lbfgs',  max_iter=1000, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)
logReg3 = LogisticRegression(solver='sag',  max_iter=1000, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)
logReg4 = LogisticRegression(solver='saga',  max_iter=1000, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)

cv_resultsv = cross_validate(logReg, dataX, dataY, cv=10, n_jobs=-1)
cv_resultsv2 = cross_validate(logReg2, dataX, dataY, cv=10, n_jobs=-1)
cv_resultsv3 = cross_validate(logReg3, dataX, dataY, cv=10, n_jobs=-1)
cv_resultsv4 = cross_validate(logReg4, dataX, dataY, cv=10, n_jobs=-1)

maxLog1 = np.mean(cv_resultsv['test_score'])
maxLog2 = np.mean(cv_resultsv2['test_score'])
maxLog3 = np.mean(cv_resultsv3['test_score'])
maxLog4 = np.mean(cv_resultsv4['test_score'])

plt.bar([1, 2, 3, 4], [maxLog1, maxLog2, maxLog3, maxLog4])
plt.xlabel('Logistic Regression Solvers')
plt.ylabel('Cross-validation Accuracy')
plt.xticks([1, 2, 3, 4], ['newton-cg', 'lbfgs', 'sag', 'saga'], rotation = 30)
plt.ylim(0.5483, 0.5484)
plt.show()



# SAGA with Elasticnet

testarrayLog = np.zeros(21)
logX = np.zeros(21)

for i in range(0,21):
    tmp = i/20
    clf = LogisticRegression(penalty = 'elasticnet', solver='saga',  max_iter=1000, l1_ratio = tmp, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayLog[i] = np.mean(cv_resultstest['test_score'])
    logX[i] = tmp

maxNet = np.amax(testarrayLog)
maxNetLoc = np.where(testarrayLog == np.amax(testarrayLog))
maxNetRat = np.divide(maxNetLoc,20)

plt.plot(logX,testarrayLog)
plt.xlabel('L1-L2 Ratio')
plt.ylabel('Cross-validation Accuracy')
plt.show()

# Log varying regularization param

testarrayLogC = np.zeros(11)
logX = np.zeros(11)

for i in range(0,11):
    tmp = (i+1)/4
    clf = LogisticRegression(solver='saga',  max_iter=1000, C = tmp, multi_class = 'multinomial', tol=1e-5, n_jobs=-1)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayLogC[i] = np.mean(cv_resultstest['test_score'])
    logX[i] = tmp

maxRegu = np.amax(testarrayLogC)
maxReguLoc = np.where(testarrayLogC == np.amax(testarrayLogC))
maxReguRat = np.divide(maxReguLoc[0]+1,4)

bestLog = LogisticRegression(solver='saga',  max_iter=1000, C = maxReguRat[0], multi_class = 'multinomial', tol=1e-5, n_jobs=-1)

# PLOT
    
plt.plot(logX,testarrayLogC)
plt.xlabel('Inverse Regularization Strength')
plt.ylabel('Cross-validation Accuracy')
plt.show()


    
# SUPPORT VECTOR MACHINE

supvec = SVC(kernel='linear', probability=True, tol=1e-4)
cv_resultsa = cross_validate(supvec, dataX, dataY, cv=10, n_jobs=-1)

maxVec = np.mean(cv_resultsa['test_score'])

# Decision Tree Classifier, Gini

testarrayGini = np.zeros(101)
logX = np.zeros(101)

for i in range(0,101):
    tmp = (i+1)*5
    clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth = tmp)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayGini[i] = np.mean(cv_resultstest['test_score'])
    logX[i] = tmp
    
bestGini = DecisionTreeClassifier(random_state=0, criterion='gini')
cv_resultspg = cross_validate(bestGini, dataX, dataY, cv=10, n_jobs=-1)
pureGiniAcc = np.mean(cv_resultspg['test_score'])

maxGini = np.amax(testarrayGini)
maxGiniLoc = np.where(testarrayGini == np.amax(testarrayGini))
maxGiniDep = (maxGiniLoc[0]+1)*5



# Decision Tree Classifier, Entropy

testarrayEntropy = np.zeros(101)
logX2 = np.zeros(101)

for i in range(0,101):
    tmp = (i+1)*5
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth = tmp)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayEntropy[i] = np.mean(cv_resultstest['test_score'])
    logX2[i] = tmp
    
bestEn = DecisionTreeClassifier(random_state=0, criterion='entropy')
cv_resultsen = cross_validate(bestEn, dataX, dataY, cv=10, n_jobs=-1)
pureEnAcc = np.mean(cv_resultsen['test_score'])
    
maxEn = np.amax(testarrayEntropy)
maxEnLoc = np.where(testarrayEntropy == np.amax(testarrayEntropy))
maxEnDep = (maxEnLoc[0]+1)*5


# PLOT
plt.plot(logX2,testarrayEntropy, 'b', label='Entropy')
plt.axhline(pureEnAcc, c='r', label='Pure Entropy')
plt.plot(logX,testarrayGini, 'g', label='Gini')
plt.axhline(pureGiniAcc, c='m', label='Pure Gini')
plt.xlabel('Tree Depth')
plt.ylabel('Cross-validation Accuracy')
plt.legend()
plt.show()

    
# MULTINOMIAL NB

testarrayMNB = np.zeros(81)
logX = np.zeros(81)

for i in range(0,81):
    tmp = i/40
    clf = ComplementNB(alpha=tmp)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayMNB[i] = np.mean(cv_resultstest['test_score'])
    logX[i] = tmp
    
    

maxMNB = np.amax(testarrayMNB)
maxMNBLoc = np.where(testarrayMNB == np.amax(testarrayMNB))
maxMNBalpha = np.divide(maxMNBLoc,40)

bestMNB = ComplementNB(alpha=maxMNBalpha[0,0])

# PLOT
plt.plot(logX,testarrayMNB)
plt.xlabel('Smoothing Parameter')
plt.ylabel('Cross-validation Accuracy')
plt.show()



# RANDOM FOREST

testarrayRF = np.zeros(21)
logX = np.zeros(21)

for i in range(0,21):
    tmp = (i*2)+1
    clf = RandomForestClassifier(n_estimators=tmp, random_state=0, n_jobs=-1)
    cv_resultstest = cross_validate(clf, dataX, dataY, cv=10, n_jobs=-1)
    testarrayRF[i] = np.mean(cv_resultstest['test_score'])
    logX[i] = tmp

maxRF = np.amax(testarrayRF)
maxRFLoc = np.where(testarrayRF == np.amax(testarrayRF))
maxRFEst = maxRFLoc[0]+1

bestRF = RandomForestClassifier(n_estimators=maxRFEst, random_state=0, n_jobs=-1)

# PLOT
plt.plot(logX,testarrayRF)
plt.xlabel('Estimator Count')
plt.ylabel('Cross-validation Accuracy')
plt.show()


# All Plots
plt.bar([1, 2, 3, 4, 5],[maxRegu, maxVec, maxGini, maxMNB, maxRF])
plt.xlabel('Max Model Accuracy')
plt.ylabel('Cross-validation Accuracy')
plt.xticks([1, 2, 3, 4, 5], ['Logistic Regression', 'SVC', 'Decision Tree', 'CNB', 'Random Forest'], rotation = 30)
plt.show()

# VOTING CLASSIFIERS (WIP)
    
vote1 = VotingClassifier(estimators=[('SVC', supvec), ('Log', bestLog), ('RF', bestRF)],
                         voting='soft', weights=[2, 2, 1])

vote2 = VotingClassifier(estimators=[('SVC', supvec), ('log', bestLog)],
                         voting='soft', weights=[1,1])

vote3 = VotingClassifier(estimators=[('CNB', bestMNB), ('Log', bestLog), ('SVC', supvec)],
                         voting='soft', weights=[2,2,1])

vote4 = VotingClassifier(estimators=[('CNB', bestMNB), ('Log', bestLog)],
                         voting='soft', weights=[1,1])

vote5 = VotingClassifier(estimators=[('CNB', bestMNB), ('Log', supvec)],
                         voting='soft', weights=[2,1])



cv_resultsVC1 = cross_validate(vote1, dataX, dataY, cv=10, n_jobs=-1)

cv_resultsVC2 = cross_validate(vote2, dataX, dataY, cv=10, n_jobs=-1)

cv_resultsVC3 = cross_validate(vote3, dataX, dataY, cv=10, n_jobs=-1)

cv_resultsVC4 = cross_validate(vote4, dataX, dataY, cv=10, n_jobs=-1)

cv_resultsVC5 = cross_validate(vote5, dataX, dataY, cv=10, n_jobs=-1)


maxVote1 = np.mean(cv_resultsVC1['test_score'])
maxVote2 = np.mean(cv_resultsVC2['test_score'])
maxVote3 = np.mean(cv_resultsVC3['test_score'])
maxVote4 = np.mean(cv_resultsVC4['test_score'])
maxVote5 = np.mean(cv_resultsVC5['test_score'])

maxVotes = np.amax([maxVote1, maxVote2, maxVote3, maxVote4, maxVote5])

