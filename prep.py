#!/usr/bin/env python3

import pandas as pd
import os, scipy.stats
import numpy as np

def Misc():    
    #Encode categorical variables as integers rather than using onehot or dummy variables- do on a column-by-column basis
    demow['ETHNIC']=pd.Categorical(demow['ETHNIC']).codes

    #pivots table
    d60p= d60.pivot(index='PATIENT',columns='TESTS',values='VALN')

    #drops duplicates if both column 1 AND column 2 have the same row value
    lab=lab.drop_duplicates(subset=['PATIENT', 'LPARM'], keep='first')
    
    #Encodes variables
    lab1['ETHNIC']=pd.Categorical(lab1['ETHNIC']).codes
    
    #pivot each column of lab to its own table (lab1, lab2, etc)
    lab1= lab.pivot(index='PATIENT',columns='LPARM',values='LVALN')
    
    #relabels columns so join works
    lab1.columns='lab1-'+lab1.columns
    
    #joins tables 

    labs=lab1.join([lab2, lab3])

    #Descriptive statistics: central tendency, dispersion and shape
    dfs2b['HAMD Total'].describe()

    #Returns vals in df1 that are also in df2
    np.intersect1d(df1['PATIENT'],df2['PATIENT'])

    #Returns vals in df1 that are not in df2
    np.setdiff1d(df1['PATIENT'],df2['PATIENT'])
    
    #Quick way to encode/binarize
    df.set_index('PATIENT')
    df['ones']=1
    df.pivot(columns='MEDGNX', values='ones')
    df=df.fillna(value=0)

    #Recode a variable
    df['Severity of illness']=df['Severity of illness'].replace(to_replace='Borderline ill', value='Borderline mentally ill')

    #Quick way to sample a class to even out classes
    df['HAMD 1=REMIT'].value_counts()
    df0=df.loc[df['HAMD 1=REMIT']==0]
    df1=df.loc[df['HAMD 1=REMIT']==1]
    dfs=df0.sample(n=1620) 
    dflist=[df1, dfs] 
    dfc=pd.concat(dflist)

    #Cgi ordered categories, -1 indicates NaN
    df=df.set_index('patient')
    cat=['Normal, not at all ill',
    'Borderline mentally ill',
    'Mildly ill',  
    'Moderately ill',
    'Markedly ill',
    'Severely ill',
    'Among the most extremely ill']
    df['CGI SEVERITY']=pd.Categorical(df['Severity of illness'], categories=cat).codes       
    
    df[df['CGI SEVERITY']==-1]
    >>>output: 89CHQS
    df=df.drop(['89CHQS'])
    
    #demow ordered sex/race
    cat=['Male', 'Female']
    
    #split off a variable
    dftherdur=df['THERDUR'].to_frame()
    del df['THERDUR']
    
    #ethnicity ordering- alphabetical like NIH
    eth=['American Indian or Alaska Native',
     'Asian',  
     'Black or African American',
     'Hispanic or Latino',
     'Native Hawaiian or Other Pacific Islander',
     'Other',
     'White']

    #sex ordering- alphabetical
    sex=['Female','Male']
    
    return


def Labeler():
    hamd=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/303-data/deid_hamd17a.csv')
    df['HAMD 1=REMIT']=np.where(df['HAMD-17 questions Total score derived']<=7, 1, 0)

            
    df.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/combined-study/class-labels.csv', index_label='PATIENT')
        
    return


def GroupDefiner():
    labels=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-d60-placebo-remitters.csv', encoding='utf-8').set_index('PATIENT').sort_index()
    placebos=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/placebo-patients.csv').set_index('PATIENT').sort_index()
    therapy= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/therapy-60-completed.csv').set_index('PATIENT').sort_index()
    
    placebos=placebos[placebos['TPNAME']=='Placebo']
    
    therapy=therapy[therapy['THERDUR>=60']==1]   
    
    final= labels.join([placebos, therapy], how='inner')
    
    del final['TPNAME']
    del final['THERDUR>=60']
    
    final.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/labels-final.csv', index_label='PATIENT')

    return
    
    
def Homeopathy():
    #Cuts all tables to subjects in labels-final
    patients=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-final.csv', encoding='utf-8').set_index('PATIENT').index
    
    path= '/media/james/ext4data1/current/projects/pfizer/303-data-baseline/'
    csvs= os.listdir(path)
    for i in csvs:
        a= pd.read_csv(path+i)
        b= a[a['PATIENT'].isin(patients)]
        b= b.set_index('PATIENT')
        b.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/303-data-baseline/cut-'+str(i), index_label='PATIENT')
    
    return

#>>>
#NOW PIVOT/CUT/ENCODE THE TABLES DOWN MANUALLY
#>>>

def Binarizer():
    #Use if you're making binarized variables
    csv= ['deid_adverse', 'deid_aemeddra', 'deid_medhist', 'deid_medhist2', 'deid_nsmed', 'deid_othtrt']
    for i in csv:
        info=pd.read_csv('/media/james/ext4data1/current/projects/pfizer/3151A1-303-csv/'+str(i)+'.csv', encoding='utf-8')
        a= info.set_index(['PATIENT'])
        b= pd.get_dummies(a)        
        d= {}
        for j in list(set(b.index)):
            d[j]= b.loc[j].values.flatten()
        
        maxlen=len(d[max(d, key=lambda k: len(d[k]))])            
        for m in d:
            d[m]=np.append(d[m], [0]*(maxlen-len(d[m])))        
        d= pd.DataFrame.from_dict(d, orient='index')
        d.columns=list(b.columns)*scipy.stats.mode(b.index).count[0]       
        d.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/vecs/vecs_'+str(i)+'.csv', index_label='PATIENT')
        #this gives a dataframe with all variables binarized 

    return


def Harvester():
    '''Because it's a combine. Aha. Ha.'''
    #But seriously, joins all tables together by patient row
    
    info= pd.read_csv('/media/james/ext4data1/current/projects/pfizer/labels-d60-placebo-remitters.csv', encoding='utf-8').set_index('PATIENT').drop('GROUPLABEL', axis=1)
    
    path= '/media/james/ext4data1/current/projects/pfizer/303-data-baseline-final/'
    csvs= os.listdir(path)    
    for i in csvs:
        a=pd.read_csv(path+i).set_index('PATIENT')
        info=info.join(a, how='inner')
    
    info.to_csv(path_or_buf='/media/james/ext4data1/current/projects/pfizer/joined-vecs.csv', index_label='PATIENT')
    
    return

   
