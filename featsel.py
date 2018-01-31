#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import copy, pickle
import numpy as np

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.

def InnerFeats():
    with open('/media/james/ext4data1/current/projects/pfizer/icv.pickle','rb') as f: icv=pickle.load(f)       
    X_train= icv['X_train']
    X_test= icv['X_test']
    y_train= icv['y_train']
    y_test= icv['y_test']
    train_indices= icv['train_indices']
    test_indices= icv['test_indices']
    
    folds= len(icv['X_train'])
    feats=[[0]]*folds
    
    skb= SelectKBest()
    #sfm=SelectFromModel(LassoCV())
    #doing second cut using rfecv to get correct # within skb
    rfe= RFECV(RandomForestClassifier(), step=1, n_jobs=3)
    
    #LOOK HERE
    #COUPLE MORE THINGS TO TRY
    #SKLEARN LASSOLARSIC SET WITH BIC MIGHT SET A MODEL WITH HIGHER ALPHA (HIGHER PENALIZATION), AND CREATE BETTER FEATURES
    for i in range(folds):
        print(i+1)
        subjects=len(X_train[i])        
        skb.fit(X_train[i], y_train[i])
        X_train_skb= skb.transform(X_train[i])
        X_test_skb= skb.transform(X_test[i])

        #rfe.fit(X_train_skb, y_train[i])
        #feats[i]=rfe.get_support(indices=True)
        #X_train_feats= rfe.transform(X_train_skb)
        #X_test_feats= rfe.transform(X_test_skb)
        
        X_train[i]= np.array(X_train_skb)
        X_test[i]= np.array(X_test_skb)
        #X_train[i]= np.array(X_train_feats)
        #X_test[i]= np.array(X_test_feats)
    
    featdict={'Feature Indices':feats, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'train_indices':train_indices, 'test_indices':test_indices}
        
    with open('/media/james/ext4data1/current/projects/pfizer/icvfeats.pickle','wb') as f: pickle.dump(featdict, f, pickle.HIGHEST_PROTOCOL)
        
    return    

def OuterFeats():
    with open('/media/james/ext4data1/current/projects/pfizer/ocv.pickle','rb') as f: ocv=pickle.load(f)       
    X_train= ocv['X_train']
    X_test= ocv['X_test']
    y_train= ocv['y_train']
    y_test= ocv['y_test']
    train_indices= ocv['train_indices']
    test_indices= ocv['test_indices']
    
    folds= len(ocv['X_train'])
    feats=[[0]]*folds
    
    for i in range(folds):
        subjects=len(X_train[i])
        skb= SelectKBest()
        skb.fit(X_train[i], y_train[i])
        feats[i]=skb.get_support(indices=True)
        X_train_feats= skb.transform(X_train[i])
        X_test_feats= skb.transform(X_test[i])
        X_train[i]= np.array(X_train_feats)
        X_test[i]= np.array(X_test_feats)
    
    featdict={'Feature Indices':feats, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'train_indices':train_indices, 'test_indices':test_indices}
        
    with open('/media/james/ext4data1/current/projects/pfizer/ocvfeats.pickle','wb') as f: pickle.dump(featdict, f, pickle.HIGHEST_PROTOCOL)
        
    return
