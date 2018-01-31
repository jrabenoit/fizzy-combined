#!/usr/bin/env python3 

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pprint, itertools, pickle, random, statistics

#Because we are sampling with replacement, we don't need to worry about the program picking all subjects each time... some may be picked more than once, and the total number of samples will be equal to the number of subjects.

def Bill():
    otr=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/outer_test_results.csv').set_index('subjects')
    
    #Per subject accuracy  
    acc= otr['scores']*100
    n= len(otr.index)
    runs= 10000

    distribution= []
    for i in range(runs):        
        sample= np.random.choice(acc, n,replace=True) 
        sample_mean= sum(sample)/len(sample)
        distribution.append(sample_mean)
    
    dist_mean= sum(distribution)/len(distribution)
    p_value= sum(i<=52.77 for i in distribution)/runs
    
    print('{} runs, {} samples per run'.format(len(distribution), n))
    print('distribution mean: {}%'.format(dist_mean))
    print('p-value: {}'.format(p_value))
    
    bootstrap_results= {'samples per run': n, 
                        'runs': 10000, 
                        'distribution mean': dist_mean, 
                        'p-value': p_value
                        }
    
    bdf= pd.DataFrame.from_dict(bootstrap_results, orient='index')
    
    binner=np.digitize(distribution, np.array(range(0,101)))
    plt.plot([52.77,52.77],[0,list(binner).count(statistics.mode(binner))],'-r',lw=2)
    plt.hist(distribution, bins=list(range(0,101)))
    plt.show()
    
    bdf.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/bootstrap_results.csv')
    
    return
