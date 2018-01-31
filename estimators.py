#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, neural_network


#Quick note: feature_importances_ can be used with random forest etc. to generate feature importance lists

def InnerFolds():
    with open('/media/james/ext4data1/current/projects/pfizer/icvfeats.pickle','rb') as f: icv=pickle.load(f)
    patients= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-final.csv', encoding='utf-8').set_index('PATIENT')
    
    folds= len(icv['X_train'])
    
    
    rf= ensemble.RandomForestClassifier(n_estimators=201, criterion='entropy', max_depth=2)
    et= ensemble.ExtraTreesClassifier(n_estimators=167, max_depth=2)
    kn= neighbors.KNeighborsClassifier(n_neighbors=35, weights='distance')
    nb= naive_bayes.GaussianNB()
    nn= neural_network.MLPClassifier(hidden_layer_sizes=(122,),solver='sgd', max_iter=2000)
    dt= tree.DecisionTreeClassifier(max_depth=2, splitter='random')
    ls= svm.LinearSVC(penalty='l1', dual=False, C=0.075)
    gb= ensemble.GradientBoostingClassifier()
    
    ab= ensemble.AdaBoostClassifier(base_estimator= rf, learning_rate=0.9)
    vc= ensemble.VotingClassifier(estimators=[('rf', rf),('kn', kn),('et',et)])
    bc= ensemble.BaggingClassifier(base_estimator=rf, n_estimators=10)
    
    est= {'randomforest': rf,
          'extratrees': et,
          'kneighbors': kn,
          'naivebayes': nb,
          'decisiontree': dt,
          'linearsvc': ls,
          'adaboost': ab,
          'neuralnet': nn,
          'voting': vc,
          'hobbitses': bc,
          'gboost': gb
          }
   
    train_results= {'fold':[], 'estimator':[], 'subjects':[], 
                    'labels':[], 'predictions':[], 'scores':[], 
                    'attempts':[]
                    }
                    
    test_results= {'fold':[], 'estimator':[], 'subjects':[], 
                   'labels':[], 'predictions':[], 'scores':[], 
                   'attempts':[]
                   }
    
    for i in range(folds):
        print(i+1)
        X_train= icv['X_train'][i]
        X_test= icv['X_test'][i]
        y_train= icv['y_train'][i]
        y_test= icv['y_test'][i]        
        train_ids= patients.index[icv['train_indices'][i]]
        test_ids= patients.index[icv['test_indices'][i]]
        
        for j,k in zip(est.keys(), est.values()):           
            k.fit(X_train, y_train)
            
            predict_train= k.predict(X_train)
            train_scores= [1 if x==y else 0 for x,y in zip(y_train, predict_train)]            
            train_results['fold'].extend([i+1]*len(X_train))
            train_results['estimator'].extend([j]*len(X_train))
            train_results['subjects'].extend(train_ids)
            train_results['labels'].extend(y_train)
            train_results['predictions'].extend(predict_train)
            train_results['scores'].extend(train_scores)
            train_results['attempts'].extend([1]*len(X_train))

            predict_test= k.predict(X_test)
            test_scores= [1 if x==y else 0 for x,y in zip(y_test, predict_test)]         
            test_results['fold'].extend([i+1]*len(X_test))
            test_results['estimator'].extend([j]*len(X_test))
            test_results['subjects'].extend(test_ids)
            test_results['labels'].extend(y_test)
            test_results['predictions'].extend(predict_test)
            test_results['scores'].extend(test_scores)
            test_results['attempts'].extend([1]*len(X_test))

    train_df=pd.DataFrame.from_dict(train_results).set_index('subjects')
    test_df=pd.DataFrame.from_dict(test_results).set_index('subjects')
    
    train_df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/inner_train_results.csv')
    test_df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/inner_test_results.csv')
    
    trd= train_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)
    pmax= trsum.idxmax(axis=1)
    print('\nBest train: {}\n'.format(pmax))

    ted= test_df.groupby('estimator').sum()
    tesum= (ted['scores']/ted['attempts'])*100
    print(tesum)
    pmax= tesum.idxmax(axis=1)
    print('\nBest test: {}\n'.format(pmax))
    
    return

def OuterFolds():
    with open('/media/james/ext4data1/current/projects/pfizer/ocvfeats.pickle','rb') as f: ocv=pickle.load(f)
    patients= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-final.csv', encoding='utf-8').set_index('PATIENT')
    
    folds= len(ocv['X_train'])

    rf= ensemble.RandomForestClassifier()
    et= ensemble.ExtraTreesClassifier()
    kn= neighbors.KNeighborsClassifier()
    nb= naive_bayes.GaussianNB()
    nn= neural_network.MLPClassifier()
    dt= tree.DecisionTreeClassifier()
    ls= svm.LinearSVC()
    gb= ensemble.GradientBoostingClassifier()
    
    ab= ensemble.AdaBoostClassifier()
    vc= ensemble.VotingClassifier(estimators=[('rf', rf),('kn', kn),('et',et)])
    bc= ensemble.BaggingClassifier(base_estimator=rf, n_estimators=100)
    
    est= {#'randomforest': rf,
          #'extratrees': et,
          #'kneighbors': kn,
          #'naivebayes': nb,
          #'decisiontree': dt
          'linearsvm': ls,
          #'adaboost': ab
          #'neuralnet': nn,
          #'voting': vc
          #'hobbitses': bc
          #'gboost': gb
          }
   
    train_results= {'fold':[], 'estimator':[], 'subjects':[], 
                    'labels':[], 'predictions':[], 'scores':[], 
                    'attempts':[]
                    }
                    
    test_results= {'fold':[], 'estimator':[], 'subjects':[], 
                   'labels':[], 'predictions':[], 'scores':[], 
                   'attempts':[]
                   }
    
    for i in range(folds):
        X_train= ocv['X_train'][i]
        X_test= ocv['X_test'][i]
        y_train= ocv['y_train'][i]
        y_test= ocv['y_test'][i]        
        train_ids= patients.index[ocv['train_indices'][i]]
        test_ids= patients.index[ocv['test_indices'][i]]
        
        for j,k in zip(est.keys(), est.values()):
            k.fit(X_train, y_train)
            
            predict_train= k.predict(X_train)
            train_scores= [1 if x==y else 0 for x,y in zip(y_train, predict_train)]            
            train_results['fold'].extend([i+1]*len(X_train))
            train_results['estimator'].extend([j]*len(X_train))
            train_results['subjects'].extend(train_ids)
            train_results['labels'].extend(y_train)
            train_results['predictions'].extend(predict_train)
            train_results['scores'].extend(train_scores)
            train_results['attempts'].extend([1]*len(X_train))

            predict_test= k.predict(X_test)
            test_scores= [1 if x==y else 0 for x,y in zip(y_test, predict_test)]         
            test_results['fold'].extend([i+1]*len(X_test))
            test_results['estimator'].extend([j]*len(X_test))
            test_results['subjects'].extend(test_ids)
            test_results['labels'].extend(y_test)
            test_results['predictions'].extend(predict_test)
            test_results['scores'].extend(test_scores)
            test_results['attempts'].extend([1]*len(X_test))

    train_df=pd.DataFrame.from_dict(train_results).set_index('subjects')
    test_df=pd.DataFrame.from_dict(test_results).set_index('subjects')
    
    train_df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/outer_train_results.csv')
    test_df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/outer_test_results.csv')

    trd= train_df.groupby('estimator').sum()
    trsum= (trd['scores']/trd['attempts'])*100
    print(trsum)
    pmax= trsum.idxmax(axis=1)
    print('\nBest train: {}\n'.format(pmax))

    ted= test_df.groupby('estimator').sum()
    tesum= (ted['scores']/ted['attempts'])*100
    print(tesum)
    pmax= tesum.idxmax(axis=1)
    print('\nBest test: {}\n'.format(pmax))
    
    return
