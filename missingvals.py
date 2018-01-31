#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values

def Impute():
    #For sparse: Impute missing vals by most_frequent, scales as sparse matrix
    #For denser: Impute by median, scale by standard scaler
    info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/joined-vecs.csv', encoding='utf-8').set_index('PATIENT')
    
    #first, remove columns with NaN values
    info= info.dropna(axis='columns')
    
    #tested axes- we want axis 0 to impute down a column
    #Using mode because of sparsity- if 99% of values are 0, most likely that this will be a 0 too.
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    X= imp.fit_transform(info)
    
    #using a sparse data scaler due to the number of zeros from binarized variables
    ss= StandardScaler()
    mas= MaxAbsScaler()
    mms= MinMaxScaler() #Gets features to a 0-1 range
    
    X= ss.fit_transform(X)
    X= mms.fit_transform(X)
    
    labels=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-final.csv', encoding='utf-8').set_index('PATIENT')
    labels= labels.join(info, how='inner')
    labels= pd.DataFrame(index=labels.index, data=labels['GROUPLABEL'], columns=['GROUPLABEL'])
    labels.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/labels-final.csv', index_label='PATIENT')
    
    
    #str((np.count_nonzero(X)/(X.shape[0]*X.shape[1]))*100) to get density... 96% sparse 
    #Should return to the psych scales (madrs, hamd etc) and encode using OneHotEncoder, since continuous variables are expected.
    
    np.savetxt('/media/james/ext4data1/current/projects/pfizer/data.csv', X, delimiter=',')
    
    return
