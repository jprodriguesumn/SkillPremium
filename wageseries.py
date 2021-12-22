#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:55:53 2019

@author: joao
"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def FilterRegData(fulldf):
    """
    Input: full dataset (person level) with transformed variables
    Output: predicted wages for a white person for each year by
        sex,education, and experience (include gdp deflator to be used later)
    """
    
    ###Create a matrix where results will lie
    edcats = ['lhs','hsc','csm','clg','gtc']
    expcats = [5,15,25,35,45]
    ### note that having edcats first ensure the pattern of resdf is the same as our predection matrix constructed below
    resdf = pd.DataFrame(list(product(edcats,expcats)), columns=['edun','expcat'])
    years = fulldf.yearn.dropna().unique()
    print(years)

    fulldf['ftfy'] = fulldf['fulltime'] * fulldf['fullyear']

    print("############## drop observations ################")
    d = fulldf.yearn.count()
    crit = (fulldf.agely >= 16) & (fulldf.agely <= 64)
    idx = fulldf[crit].index
    fulldf = fulldf.loc[idx,:]
    d1 = fulldf.yearn.count()
    print(d-d1, " obs dropped")

    crit = (~fulldf.winc_ws.isna())
    idx = fulldf[crit].index
    fulldf = fulldf.loc[idx,:]
    d2 = fulldf.yearn.count()
    print(d1-d2, " obs dropped")

    crit = (fulldf.allocated==0)
    idx = fulldf[crit].index
    fulldf = fulldf.loc[idx,:]
    d3 = fulldf.yearn.count()
    print(d2-d3, " obs dropped")

    crit = (fulldf.selfemp==0)
    idx = fulldf[crit].index
    fulldf = fulldf.loc[idx,:]
    d4 = fulldf.yearn.count()
    print(d3-d4, " obs dropped")

    fulldf.loc[:,'rlnwinc'] = np.log(fulldf.winc_ws) + np.log(fulldf.gdp) 
    fulldf.loc[:,'exp1'] = fulldf.pexp

    dummy = pd.get_dummies(fulldf[['edun','racen']])
    fulldf = pd.concat([fulldf,dummy],axis=1)

    fulldf.loc[:,'exp2'] = (fulldf.exp1.pow(2))/100
    fulldf.loc[:,'exp3'] = (fulldf.exp1.pow(3))/1000
    fulldf.loc[:,'exp4'] = (fulldf.exp1.pow(4))/10000

    fulldf.loc[:,'e1edhsd'] = fulldf.exp1*fulldf.edun_lhs
    fulldf.loc[:,'e1edcsm'] = fulldf.exp1*fulldf.edun_csm
    fulldf.loc[:,'e1edclg'] = fulldf.exp1*np.minimum(fulldf.edun_clg + fulldf.edun_gtc,1)

    fulldf.loc[:,'e2edhsd'] = fulldf.exp2*fulldf.edun_lhs
    fulldf.loc[:,'e2edcsm'] = fulldf.exp2*fulldf.edun_csm
    fulldf.loc[:,'e2edclg'] = fulldf.exp2*np.minimum(fulldf.edun_clg + fulldf.edun_gtc,1)

    fulldf.loc[:,'e3edhsd'] = fulldf.exp3*fulldf.edun_lhs
    fulldf.loc[:,'e3edcsm'] = fulldf.exp3*fulldf.edun_csm
    fulldf.loc[:,'e3edclg'] = fulldf.exp3*np.minimum(fulldf.edun_clg + fulldf.edun_gtc,1)

    fulldf.loc[:,'e4edhsd'] = fulldf.exp4*fulldf.edun_lhs
    fulldf.loc[:,'e4edcsm'] = fulldf.exp4*fulldf.edun_csm
    fulldf.loc[:,'e4edclg'] = fulldf.exp4*np.minimum(fulldf.edun_clg + fulldf.edun_gtc,1)

    fulldf.loc[:,'tcwkwg'] = 0
    fulldf.loc[:,'tchkwg'] = 0


    #fulldf = fulldf[(fulldf.ftfy==1) & (fulldf.tcwkwg==0) & (fulldf.bcwkwgkm==0)]
    #print("female obs: ",fulldf[(fulldf.yearn==1976)&(fulldf.sexn=='female')].yearn.count())
    ### create dataframe to carry full multiyear results
    alldf_results = []
    for sex in ['male','female']:
        ## create dataframe to carry sex specific results
        sexdf_results = []
        if sex != 'both' :
            sexcrit = (fulldf.sexn==sex)
            sexidx = fulldf[sexcrit].index
            sexdf = fulldf.loc[sexidx,:]
            #sexdf = fulldf.copy()[(fulldf.sexn==sex)]

        sexdf = fulldf.copy()
        print("Number of full sex specific obs: ",sexdf.yearn.count())

        ####### year/sex specific regressions
        #for year in range(1979,1985) :
        for year in years :
            print("############################### years: ",year,"##################################")
            crit = (sexdf.yearn==year)
            idx = sexdf[crit].index
            df = sexdf.loc[idx,:]
            
            print("Number of full sex specific obs: ",df.yearn.count())
            print("############ quantities per cell ##################")
            print(df.groupby(['yearn','edun','exp1'])[['yearn']].count())

            #print(df[df.yearn==year].gdp.unique())
            #gdpyear = df[df.yearn==year].gdp.unique()[0]
            Xvars = ['edun_lhs','edun_csm','edun_clg','edun_gtc',
                     'exp1','exp2','exp3','exp4',
                     'racen_black','racen_other',
                     'e1edhsd','e1edcsm','e1edclg',
                     'e2edhsd','e2edcsm','e2edclg',
                     'e3edhsd','e3edcsm','e3edclg',
                     'e4edhsd','e4edcsm','e4edclg']
            X = df[Xvars]

            #### potentially use scikit numpy based (may be faster)
            y = df[['rlnwinc']]
            #regr = linear_model.LinearRegression(fit_intercept=True)
            #regr.fit(X, y, df.wgt)
            #predictions = regr.predict(X)
            #print(regr.coef_)

            X = sm.add_constant(X)
            model = sm.WLS(y, X,weights=df[['wgt']]).fit()
            """
            Create the average white individual based on
                5 experience categories and
                5 education levels
            We need to create a dataframe for a value for each of the predictors used in the regression
            so we can use the prediction tool from statsmodels. For each of these observations, it'll 
            spit out prediction wages based on regression coefficients. We'll consider the average person to
            fall right at the midpoint of each experience category (5,15,25,...). So will need to adjust exp1
            and all other exp1 interaction terms as defined above.
            """
            newX = [[1,1,0,0,0,5],
                    [1,1,0,0,0,15],
                    [1,1,0,0,0,25],
                    [1,1,0,0,0,35],
                    [1,1,0,0,0,45],
                    [1,0,0,0,0,5],
                    [1,0,0,0,0,15],
                    [1,0,0,0,0,25],
                    [1,0,0,0,0,35],
                    [1,0,0,0,0,45],
                    [1,0,1,0,0,5],
                    [1,0,1,0,0,15],
                    [1,0,1,0,0,25],
                    [1,0,1,0,0,35],
                    [1,0,1,0,0,45],
                    [1,0,0,1,0,5],
                    [1,0,0,1,0,15],
                    [1,0,0,1,0,25],
                    [1,0,0,1,0,35],
                    [1,0,0,1,0,45],
                    [1,0,0,0,1,5],
                    [1,0,0,0,1,15],
                    [1,0,0,0,1,25],
                    [1,0,0,0,1,35],
                    [1,0,0,0,1,45]]
            newX = pd.DataFrame(newX,columns=['cons','edun_lhs','edun_csm','edun_clg','edun_gtc','exp1'])
            newX.loc[:,'exp2'] = (newX.exp1.pow(2))/100
            newX.loc[:,'exp3'] = (newX.exp1.pow(3))/1000
            newX.loc[:,'exp4'] = (newX.exp1.pow(4))/10000

            newX.loc[:,'racen_black'] = 0
            newX.loc[:,'racen_other'] = 0
            newX.loc[:,'e1edhsd'] = newX.exp1*newX.edun_lhs
            newX.loc[:,'e1edcsm'] = newX.exp1*newX.edun_csm
            newX.loc[:,'e1edclg'] = newX.exp1*np.minimum(newX.edun_clg + newX.edun_gtc,1)
            newX.loc[:,'e2edhsd'] = newX.exp2*newX.edun_lhs
            newX.loc[:,'e2edcsm'] = newX.exp2*newX.edun_csm
            newX.loc[:,'e2edclg'] = newX.exp2*np.minimum(newX.edun_clg + newX.edun_gtc,1)
            newX.loc[:,'e3edhsd'] = newX.exp3*newX.edun_lhs
            newX.loc[:,'e3edcsm'] = newX.exp3*newX.edun_csm
            newX.loc[:,'e3edclg'] = newX.exp3*np.minimum(newX.edun_clg + newX.edun_gtc,1)

            newX.loc[:,'e4edhsd'] = newX.exp4*newX.edun_lhs
            newX.loc[:,'e4edcsm'] = newX.exp4*newX.edun_csm
            newX.loc[:,'e4edclg'] = newX.exp4*np.minimum(newX.edun_clg + newX.edun_gtc,1)

            predictions = model.predict(newX)
            #print(model.summary())
            print("predictions for sex: ",sex)
            print(predictions)
            print(model.params[0:5])
            sdf = pd.concat([resdf,predictions],axis=1)
            sdf=sdf.rename(columns = {0:'plnwkw'})
            sdf.loc[:,'yearn'] = year
            #sdf.loc[:,'gdp'] = gdpyear
            sdf.loc[:,'sexn'] = sex
            print("########## sex reg results #############")
            print(sdf)
            sexdf_results.append(sdf)
        print("########## concated data ############")
        sexdf_results = pd.concat(sexdf_results)
        print(sexdf_results)
        alldf_results.append(sexdf_results)

    fdf = pd.concat(alldf_results)

    return fdf, newX


def Aggregate_eu(df):
    kpvars = ['yearn','sexn','edun','pexp','q_lsweight']
    df = df[kpvars]

    ### get experience categories from wage regressions
    df.loc[(df.pexp >=0) & (df.pexp <=9),'expcat'] = 5
    df.loc[(df.pexp >=10) & (df.pexp <=19),'expcat'] = 15
    df.loc[(df.pexp >=20) & (df.pexp <=29),'expcat'] = 25
    df.loc[(df.pexp >=30) & (df.pexp <=39),'expcat'] = 35
    df.loc[(df.pexp >=40),'expcat'] = 45

    df = df.groupby(['yearn','sexn','edun','expcat'])[['q_lsweight']].sum()
    df=df.rename(columns = {'q_lsweight':'lswt'})
    df = df.reset_index()
    return df

def wagesSupply(wdf,sdf):

    df = pd.merge(wdf,sdf,how='left',on=['yearn','sexn','expcat','edun'])

    ### get weight across all groups by year
    s = df[['yearn','lswt']].groupby(['yearn']).sum()
    s.columns = ['t1']
    s = s.reset_index()
    df = pd.merge(df,s,how='left', on='yearn')
    df.loc[:,'normlswt'] = df.lswt/df.t1

    ### get average across all years by group
    s = df[['sexn','expcat','edun','normlswt']].groupby(['sexn','expcat','edun']).mean()
    s.columns = ['avlswt']
    s = s.reset_index()
    df = pd.merge(df,s,how='left', on=['sexn','expcat','edun'])

    df.loc[:,'rplnwkw'] = df.plnwkw #+ np.log(df.gdp) 
    df.loc[:,'rplavl'] = df.rplnwkw * df.avlswt


    return df

def ComputePremium(finaldf):
    '''
    input: final data with predicted wages by cell (year,sex,exp,edu)
    '''
    datayrn=[]
    dataexp=[]
    for i in ['m','f','both']:
        if i=='m' :
            df = finaldf[finaldf.sexn=='male']
        elif i=='f' :
            df = finaldf[finaldf.sexn=='female']
        else:
            df = finaldf

        #df.loc[:,'rplavl'] = df.rplnwkw * df.avlswt
        colhs = pd.DataFrame([['hsc','hs'],['clg','col']],
                             columns= ['edun','edcat'])
        colphs = pd.DataFrame([['hsc','hs'],['clg','col'],['gtc','col']],
                             columns= ['edun','edcat'])
        colphsp = pd.DataFrame([['lhs','hs'],['hsc','hs'],['clg','col'],['gtc','col']],
                             columns= ['edun','edcat'])

        colhsdf = pd.merge(df,colhs,how='left',on='edun')
        colphsdf = pd.merge(df,colphs,how='left',on='edun')
        colphspdf = pd.merge(df,colphsp,how='left',on='edun')
        colhsdf = colhsdf.dropna()
        colphsdf = colphsdf.dropna()
        ###col/hs
        s = colhsdf[['yearn','rplavl','avlswt','edcat']].groupby(['yearn','edcat']).sum()
        s.loc[:,'wages'] = s.rplavl/s.avlswt
        s = s.unstack('edcat')
        s.columns = [n[0] + "_" + n[1] for n in s.columns] ### joins the original variable with the categories unstacked
        s.loc[:,'clghsg_all'] = s.wages_col - s.wages_hs

        ###colp/hs
        s1 = colphsdf[['yearn','rplavl','avlswt','edcat']].groupby(['yearn','edcat']).sum()
        s1.loc[:,'wages'] = s1.rplavl/s1.avlswt
        s1 = s1.unstack('edcat')
        s1.columns = [n[0] + "_" + n[1] for n in s1.columns] ### joins the original variable with the categories unstacked
        s1.loc[:,'clphsg_all'] = s1.wages_col - s1.wages_hs
        ss = pd.merge(s[['clghsg_all']],s1[['clphsg_all']],how='inner',left_index=True, right_index=True)
        ss.loc[:,'sex'] = i

        ###col/hs
        r = colhsdf[['yearn','rplavl','avlswt','edcat','expcat']].groupby(['yearn','expcat','edcat']).sum()
        r.loc[:,'wages'] = r.rplavl/r.avlswt
        r = r.unstack('edcat')
        r.columns = [n[0] + "_" + n[1] for n in r.columns] ### joins the original variable with the categories unstacked
        r.loc[:,'clghsg_exp'] = r.wages_col - r.wages_hs

        ###colp/hs
        r1 = colphsdf[['yearn','rplavl','avlswt','edcat','expcat']].groupby(['yearn','expcat','edcat']).sum()
        r1.loc[:,'wages'] = r1.rplavl/r1.avlswt
        r1 = r1.unstack('edcat')
        r1.columns = [n[0] + "_" + n[1] for n in r1.columns] ### joins the original variable with the categories unstacked
        r1.loc[:,'clphsg_exp'] = r1.wages_col - r1.wages_hs
        rr = pd.merge(r[['clghsg_exp']],r1[['clphsg_exp']],how='inner',left_index=True, right_index=True)
        rr.loc[:,'sex'] = i

        datayrn.append(ss)
        dataexp.append(rr)

    return pd.concat(datayrn),pd.concat(dataexp)





"""

cldf = pd.read_pickle("cleaned/cleaned.pkl")
df,Xdf = FilterRegData(cldf)
df.to_pickle("cleaned/predicwags.pkl")

##### this approximately matches autor's predicted wages
dfwag = pd.read_pickle("cleaned/predicwags.pkl")
dfwagautor = pd.read_stata("/home/joao/Dropbox/autor_files/prep-wage/pred-marwg-6308.dta")

###### this also approximately matches
dfsup = pd.read_pickle("cleaned/lswts.pkl")
dfsupautor = pd.read_stata("/home/joao/Dropbox/autor_files/prep-wage/march-lswts-exp.dta")

### mege these together
fdf = wagesSupply(dfwag,dfsup)
### this has wages merged with supply
dfwagautor = pd.read_stata("/home/joao/Dropbox/autor_files/prep-wage/pred-marwg-6308.dta")  
#earnautor = pd.read_stata("/home/joao/Dropbox/autor_files/prep-wage/march-lswts-exp.dta")  
plt.figure()  
dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==1) & (dfwagautor.expcat=='45 years')].plot(kind='line')
dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==2) & (dfwagautor.expcat=='45 years')].plot(kind='line')
dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==3) & (dfwagautor.expcat=='45 years')].plot(kind='line')
dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==4) & (dfwagautor.expcat=='45 years')].plot(kind='line')
dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==5) & (dfwagautor.expcat=='45 years')].plot(kind='line')

#dfwagautor['rplnwkw'][(dfwagautor.female==0) & (dfwagautor.edcat5==2) & (dfwagautor.expcat=='45 years')].plot(kind='line')

plt.figure()
labsup = pd.read_pickle("cleaned/eudfagg.pkl")    
pt = labsup.pivot(index='yearly',columns='expcat',values='euexp_shclg') 
pt.plot(kind='line',title='labor supply')

plt.figure()    
dfy,dfe = ComputePremium(fdf) 
dfe = dfe.reset_index()  
pt = dfe[dfe.sex=='m'].pivot(index='yearly',columns='expcat',values='clghsg_exp') 
pt.plot(kind='line',title='labor supply')

#dfy = dfy.reset_index()
#dfy = dfy.set_index('yearly')
#plt.figure()
#dfy['clghsg_all'][(dfy.sex=='m')].plot(kind='line')


"""
