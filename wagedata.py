#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:40:16 2019

@author: joao
"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
#from statsmodels.datasets import grunfeld
from pandas.api.types import CategoricalDtype
#from pandas.api.types import union_categoricals

#data = grunfeld.load_pandas().data
#data.year = data.year.astype(np.int64)
#from linearmodels import PanelOLS
#etdata = data.set_index(['firm','year'])

def PrepareData(df,interval):
    """
    variables used:  YEAR, UHRSWORKLY, WKSWORK1, POPSTAT, QINCWAGE, SRCEARN, QOINCWAGE, QINCLONG, ACTNLFLY, METAREA, EDUC, SEX, HIGRADE, RACE, INDLY, CLASSWLY, FULLPART, ASECWT, OINCWAGE, INCLONGJ
    Input: raw data from march cps downloaded from IPUMS
    Output: data with various new variables
    only keep:
        2. Non-armed forces civilians who worked at least 1 week and are older than 17
        1. self employed and salaried workers
    """
    varsused = ['YEAR','AGE','STATEFIP' ,'UHRSWORKLY', 'WKSWORK1', 'POPSTAT', 'QINCWAGE', 'SRCEARN', 'QOINCWAGE', 'QINCLONG', 'ACTNLFLY', 'METAREA', 'EDUC', 'SEX', 'HIGRADE', 'RACE', 'INDLY', 'CLASSWLY', 'FULLPART', 'ASECWT', 'INCWAGE','OINCWAGE', 'INCLONGJ']

    assert set(varsused).issubset(set(df.columns))

    ### rename hour variables
    df.loc[:,'hrslyr'] = df.UHRSWORKLY
    df.loc[:,'wkslyr'] = df.WKSWORK1  #### weeks worked in main job (ues df.WKSWORKT for all jobs)
    #df.loc[:,'hrslwk'] = df.AHRSWORKT ### download this variable

    ### add popstat here
    print("######################################################################")
    print("#########keep only workers (workd age) who worked at least 1 week######")
    d1 = df.YEAR.count()

    #### get indices to keep
    crit = (df.POPSTAT==1) & (df.wkslyr>=1) & (df.wkslyr<=52) & (df.AGE>=17)
    idx = df[crit].index

    #### create new dataframe
    df = df.loc[idx,:]

    ### new count
    d2 = df.YEAR.count()
    print("dropped ",d1-d2," observations"," new size: ",d2)


    ### Allocated flag from (se earnings or wage earnings): note that oinc includes self emp income
    df.loc[df.YEAR<=1987,'allocated'] = (((df.QINCWAGE==1) | (df.QINCWAGE==2) | (df.QINCWAGE==3)) & (df.SRCEARN==1)).astype(int)
    df.loc[df.YEAR>=1988,'allocated'] = (( (df.QOINCWAGE==1) | (df.QINCLONG==1)) & (df.SRCEARN==1)).astype(int)

    ### keep those in school last year who also worked some
    df.loc[:,"studently"] = ((df.ACTNLFLY == 30) & (df.wkslyr<=49)).astype(int)

    #### topcoded ages and hours in autor's code not an issue here?

    ### create string variable with metarea codes with leading zeros
    df.loc[:,'METAREA']=df['METAREA'].apply(lambda x: '{0:0>4}'.format(x))
    df.loc[:,'met'] = df.METAREA.str[0:3]

    ### keep only met area
    #df = df[df.met != '999']

    #####################################################################
    ### education categories for composition adjusted wage
    #####################################################################
    """
    #lhs less than high school
    #hsc finished high school but no college
    #csm some college, dit not finish
    #clg finished college
    #gtc more than college, possibly grad school
    """
    lhs = [1,2,10,20,30,40,50,60,71]
    hsc = [73] ## high school diploma
    csm = [80,81,90,91,92,100]
    clg = [110,111]
    gtc = [120,121,122,123,124,125]
    df.loc[df.EDUC.isin(lhs),'edun'] = "lhs"
    df.loc[df.EDUC.isin(hsc),'edun'] = "hsc"
    df.loc[df.EDUC.isin(csm),'edun'] = "csm"
    df.loc[df.EDUC.isin(clg),'edun'] = "clg"
    df.loc[df.EDUC.isin(gtc),'edun'] = "gtc"
    edun_cats = ['lhs','hsc','csm','clg','gtc']
    edun_type = CategoricalDtype(categories=edun_cats, ordered=True)
    df.edun = df.edun.astype(edun_type)

    df.loc[(df.SEX==1),'sexn'] = 'male'
    df.loc[(df.SEX==2),'sexn'] = 'female'
    female_cats = ['female','male']
    female_type = CategoricalDtype(categories=female_cats, ordered=False)
    df.sexn = df.sexn.astype(female_type)
    #####################################################################
    ########### build experience var ####################################
    #####################################################################
    df.loc[df.EDUC.isin([1,2,10,11,12,13,14]),'edyears'] = 0
    df.loc[df.EDUC.isin([20,21,22]),'edyears'] = 6
    df.loc[df.EDUC.isin([30,31,32]),'edyears'] = 8
    df.loc[df.EDUC.isin([40]),'edyears'] = 9
    df.loc[df.EDUC.isin([50]),'edyears'] = 10
    df.loc[df.EDUC.isin([60]),'edyears'] = 11
    df.loc[df.EDUC.isin([70,71,72,73]),'edyears'] = 12
    df.loc[df.EDUC.isin([80,81]),'edyears'] = 13
    df.loc[df.EDUC.isin([90,91,92]),'edyears'] = 14
    df.loc[df.EDUC.isin([100]),'edyears'] = 15
    df.loc[df.EDUC.isin([110,111]),'edyears'] = 16
    df.loc[df.EDUC.isin([120,121]),'edyears'] = 17
    df.loc[df.EDUC.isin([122,123,124]),'edyears'] = 18
    df.loc[df.EDUC.isin([125]),'edyears'] = 18

    eddf = pd.read_csv("edcatspre92.csv")
    dp = df.YEAR.count()
    df = pd.merge(df,eddf,how='left',on='HIGRADE')
    dp1 = df.YEAR.count()
    assert dp == dp1
    df.loc[(df.YEAR <= 1991),'edyears'] = df.edyearspre92
    df.loc[(df.YEAR <= 1991),'edun'] = df.edunpre92
    #### use 7 bc this is related to last year's activities
    df.loc[:,'pexp'] = np.maximum(np.minimum(df['AGE'] - df['edyears'] - 7,df['AGE']-17),0)

    ### experience cats
    df.loc[(df.pexp >=0) & (df.pexp <=9),'grpexp'] = 'exp0_9'
    df.loc[(df.pexp >=10) & (df.pexp <=19),'grpexp'] = 'exp10_19'
    df.loc[(df.pexp >=20) & (df.pexp <=29),'grpexp'] = 'exp20_29'
    df.loc[(df.pexp >=30) & (df.pexp <=39),'grpexp'] = 'exp30_39'
    df.loc[(df.pexp >=40),'grpexp'] = 'exp40_p'
    grpexp_cats = ['exp0_9','exp10_19','exp20_29','exp30_39','exp40_p']
    grpexp_type = CategoricalDtype(categories=grpexp_cats, ordered=True)
    df.grpexp = df.pexp.astype(grpexp_type)

    ### college noncollege cats
    colvars = ["clg","gtc"]
    nocolvars = ["csm","hsc",'lhs']
    df.loc[df.edun.isin(nocolvars),'college'] = 'nocol'
    df.loc[df.edun.isin(colvars),'college'] = "col"
    college_cats = ['nocol','col']
    college_type = CategoricalDtype(categories=college_cats, ordered=True)
    df.college = df.college.astype(college_type)

    ### race cats
    df.loc[:,'racen'] = 'other'
    df.loc[(df.RACE == 100),'racen'] = "white"
    df.loc[(df.RACE == 200),'racen'] = "black"
    race_cats = ['white','black','other']
    race_type = CategoricalDtype(categories=race_cats, ordered=False)
    df.racen = df.racen.astype(race_type)

    ### data here is based on things done in previous year (ASEC sample)
    df.loc[:,'yearly'] = df.YEAR - 1

    ### create last year age variable
    df.loc[(df.AGE >= 17) & (df.AGE <= 71),'agely'] = df.AGE - 1
    df.loc[(df.AGE > 71),'agely'] = 71

    ### drop ag workers last year
    df.loc[(df.INDLY>=17) & (df.INDLY<=29) & (df.YEAR<=1982),'aggworker'] = 1
    df.loc[(df.INDLY>=10) & (df.INDLY<=31) & (df.YEAR>=1983) & (df.YEAR<=1991),'aggworker'] = 1
    df.loc[(df.INDLY>=10) & (df.INDLY<=32) & (df.YEAR>=1992) & (df.YEAR<=2002),'aggworker'] = 1
    df.loc[(df.INDLY>=170) & (df.INDLY<=280) & (df.YEAR>=2003) & (df.YEAR<=2008),'aggworker'] = 1
    df.loc[(df.INDLY>=170) & (df.INDLY<=290) & (df.YEAR>=2009) & (df.YEAR<=2013),'aggworker'] = 1
    df.loc[(df.INDLY>=170) & (df.INDLY<=290) & (df.YEAR>=2014) ,'aggworker'] = 1

    ### putting all relevant categories together (relevant for composition regressions)
    #df['all_cats'] = df['edun'] + df['pexp'].astype(int).apply(str) + df['racn']

    #print("Ensure existence of agg workers each year")
    #print(df[df.aggworker==1].YEAR.value_counts())
    ###############################################################################
    ################  cats of fulltime,fullyear,selfempworker, and wageworker ###############
    ##############################################################################
    ### self employment workers
    selemp = [13,14]
    df.loc[:,'selfemp'] = (df.CLASSWLY.isin(selemp)).astype(int)

    ### wage worker
    wagemp = [22,24,25,27,28]
    df.loc[:,'wageworker'] = (df.CLASSWLY.isin(wagemp)).astype(int)

    ######################################################
    ### keep if either a wage worker or self employed#####
    ######################################################
    print("keep if either a wage worker or self employed")
    df = df[(df.selfemp==1) | (df.wageworker==1)]
    d3 = df.YEAR.count()
    print("dropped ",d2-d3," observations"," new size: ",d3)

    ### full year worker = worked more than 35 weeks
    df.loc[:,'fullyear'] = ((df.wkslyr>=40) & (df.wkslyr<=52)).astype(int)

    ### full time = usually worked 35+ hours per week
    df.loc[:,'fulltime'] = (df.FULLPART==1).astype(int)
    #df.loc[~(df.FULLPART==1),'fulltime'] = 0

    ### generate new weights based on weeks or hourly earnings
    df.loc[:,'wgt'] = df.ASECWT
    df.loc[:,'wgt_wks'] = df.wgt * df.wkslyr
    df.loc[:,'wgt_hrs'] = df.wgt * df.wkslyr * df.hrslyr
    df.loc[:,'wgt_hrs_ft'] = df.wgt * df.hrslyr
    ################################################
    #################### earnings
    ###################################################
    """
    #1980-1987: just take incwage = lincj
    #1988-1995: incwage = lincj + oincwag
    #1996-2018: values of lincj and oincwag are changed by ipums (most probable values)
    """

    ### get topcoded limits
    ildf = pd.read_csv("inclims.csv",delimiter=",")
    df = pd.merge(df,ildf,how='left',on='YEAR')

    ###################################################
    ### before 1988 ###################################
    ###################################################
    df.loc[:,'maxwag'] = df.maxoinc + df.maxlinc
    df.loc[:,'incwagen'] = df.INCWAGE
    df.loc[(df.YEAR<=1987) & (df.INCWAGE>=df.maxwag),'incwagen'] = 1.5*df.maxwag
    df.loc[(df.YEAR<=1987),'tcwkwg'] =  (df.incwagen/df.wkslyr> df.maxwag*1.5/40).astype(int)
    df.loc[(df.YEAR<=1987),'tchrwg'] =  (df.incwagen/(df.wkslyr*df.hrslyr) > df.maxwag*1.5/(1400)).astype(int)

    ### Assign weekely wages to wageworkers
    df.loc[(df.YEAR<=1987) & (df.wageworker==1) & (df.incwagen > 0),'winc_ws'] = df.incwagen/df.wkslyr 

    ### Assign hourly wages to wageworkers
    df.loc[(df.YEAR<=1987) & (df.wageworker==1) & (df.incwagen > 0),'hinc_ws'] = df.winc_ws/df.hrslyr

    ###################################################
    ### between 1988 and 1995 #########################
    ###################################################
    df.loc[:,'oincwagen'] = df.OINCWAGE
    df.loc[:,'inclongjn'] = df.INCLONGJ
    df.loc[(df.YEAR>=1988) & (df.YEAR<=1995) & (df.oincwagen>=df.maxoinc) & (~df.oincwagen.isna()),'oincwagen'] = 1.5*df.maxoinc
    df.loc[(df.YEAR>=1988) & (df.YEAR<=1995) & (df.inclongjn>=df.maxlinc) & (~df.inclongjn.isna()),'inclongjn'] = 1.5*df.maxlinc
    df.loc[(df.YEAR>=1988) ,'tcwkwg'] =  ((df.oincwagen + df.inclongjn)/df.wkslyr > (df.maxoinc + df.inclongjn)*1.5/40).astype(int)
    df.loc[(df.YEAR>=1988) ,'tchrwg'] =  ((df.oincwagen + df.inclongjn)/(df.wkslyr*df.hrslyr) > (df.maxoinc + df.inclongjn)*1.5/1400).astype(int)

    ### Assign weekely wages to wageworkers
    df.loc[(df.YEAR>=1988) & (df.wageworker==1) & (df.oincwagen + df.inclongjn > 0),'winc_ws'] = (df.oincwagen + df.inclongjn)/df.wkslyr

    ### Assign hourly wages to wageworkers
    df.loc[(df.YEAR>=1988) & (df.wageworker==1) & (df.oincwagen + df.inclongjn > 0),'hinc_ws'] = (df.oincwagen + df.inclongjn)/df.hrslyr

    ####################################################
    ### do nothing for values swapped by IPUMS #########
    ####################################################


    ####################################################
    #### Create flags based on Katz and Murphy  cats ###
    ####################################################
    ### get deflator
    gdpdata = pd.read_csv("GDPDEF.csv")
    gdpdata = gdpdata[gdpdata.month==4][['YEAR','GDPDEF']]
    #### when I put these numbers, it matches a describe of various flags indicate data are the same.
    gdpdata.loc[:,'gdp'] = 100/gdpdata['GDPDEF']
    df.loc[:,'YEAR'] = df.YEAR - 1  #### make year relevant to previous year
    df = pd.merge(df,gdpdata,how='left',on='YEAR')
    gdp1982 = gdpdata[gdpdata.YEAR==1982].GDPDEF.unique()[0]

    df.loc[:,'bcwkwg'] = (df.winc_ws*df.gdp < (40*(100/gdp1982))).astype(int)
    df.loc[:,'bchrwg'] = (df.winc_ws*df.gdp/df.hrslyr < (1*(100/gdp1982))).astype(int)
    df.loc[:,'bcwkwgkm'] = (df.winc_ws*df.gdp < (67*(100/gdp1982))).astype(int)
    df.loc[:,'bchrwgkm'] = (df.winc_ws*df.gdp/df.hrslyr< (1.675*(100/gdp1982))).astype(int)

    #cpidata = pd.read_csv("CPIAUCSL.csv")
    #cpidata = cpidata[cpidata.month==6][['YEAR','cpi']]
    #cpidata.loc[:,'YEAR'] = cpidata.YEAR + 1  #### make year relevant to previous year
    #cpidata.loc[:,'cpi2000'] = cpidata[cpidata.YEAR==2000].cpi.unique()[0]/cpidata['cpi']
    #df = pd.merge(df,cpidata,how='left',on='YEAR')

    #print(df.groupby('yearly').mean()[['wgt','winc_ws','wgt_wks','wgt_hrs','allocated']])
    #print(df.groupby('yearly').mean()[['tcwkwg','tchrwg','bcwkwg']])
    #print(df.groupby('yearly').mean()[['bchrwg','bchrwgkm','bcwkwgkm']])

    #### Define new year variable based on interval
    yrmin = df.yearly.unique().min()
    yrmax = df.yearly.unique().max()
    yearsrangeint = list(range(yrmin+interval-1,yrmax,interval))
    yearsrange1 = range(yrmin,max(list(yearsrangeint))+1)
    nyears = np.repeat(yearsrangeint,interval)

    dfyears = pd.DataFrame({'yearn' : nyears, 'yearly' : yearsrange1})
    df = pd.merge(df,dfyears,how='left',on='yearly')
    
    variables = ['yearly','yearn','met','STATEFIP','allocated','studently',
                 'wgt','wgt_hrs_ft','wgt_hrs','wgt_wks','wkslyr','hrslyr',
                 'edun','sexn','edyears','pexp','grpexp','college','racen','agely',
                 'aggworker','selfemp','wageworker','fullyear','fulltime','tcwkwg','tchrwg',
                 'winc_ws','hinc_ws','gdp','GDPDEF','bcwkwg','bchrwg','bcwkwgkm','bchrwgkm','SRCEARN']
    df = df[variables]

    return df

def LaborSupply(df):
    """
    Intput: data from PrepareData with newly defined variables
    Output: A dataset at the "cell level", i.e. at the level of the disaggregation we chose fort he decomposition of wages, (edun,sexn,pexp)
    """

    #### tests
    d1 = df.yearly.count()

    #### get indices
    crit = (df.agely>=16) & (df.agely<=64)
    idx = df[crit].index

    #### create new dataframe
    df = df.loc[idx,:]

    #### check size
    d2 = df.yearly.count()
    print(d1-d2," observations dropped")

    df['ftfy'] = df.fulltime * df.fullyear


    ### only keep ftfy workers not those making less than 1.67 in 1982
    crit = (df.ftfy==0) | (df.bcwkwgkm==1)
    df.loc[crit,'winc_ws'] = np.nan    #making nan appears to behave similar to stata. Removes obs from count, ignores in stats
    crit = (df.ftfy==0) | (df.bchrwgkm==1)
    df.loc[crit,'hinc_ws'] = np.nan

    dummies = pd.get_dummies(df[['edun','racen','sexn']])
    df = pd.concat([df,dummies],axis=1)

    ##### grab education categories
    educats = [s for s in df.columns if 'edun_' in s]
    #print(educats)
    ##### create interaction term with potential experience
    for educat in educats:
        df.loc[:,educat+'pexp'] = df[educat] * df.pexp

    ##### create square term of experience
    df.loc[:,'pexpsq'] = df.pexp.pow(2)
    for educat in educats:
        df.loc[:,educat+'pexpsq'] = df[educat] * df.pexpsq

    df.loc[:,'rwinc'] = df.winc_ws*df.gdp
    df.loc[:,'lnwinc'] = np.log(df.winc_ws)
    df.loc[:,'rlnwinc'] = df.lnwinc + np.log(df.gdp)
    df.loc[:,'rhinc'] = df.hinc_ws*df.gdp
    df.loc[:,'lnhinc'] = np.log(df.hinc_ws)
    df.loc[:,'rlnhinc'] = df.lnhinc + np.log(df.gdp)

    df.loc[:,'q_obs'] = 1
    df.loc[:,'q_weight'] = df.wgt
    df.loc[:,'q_lsweight'] = df.wgt*df.wkslyr
    df.loc[:,'q_lshrsweight'] = df.wgt*df.wkslyr*df.hrslyr


    #### obs to drop
    df.loc[:,'p_obs'] = 1
    crit = (df.winc_ws.isna()) | (df.selfemp == 1) | (df.allocated==1)
    df.loc[crit,'p_obs'] = 0
    df.loc[:,'p_weight'] = df.wgt
    df.loc[crit,'p_weight'] = 0
    df.loc[:,'p_lsweight'] = df.wgt*df.wkslyr
    df.loc[crit,'p_lsweight'] = 0

    collapse_vars = ['yearn','edun','pexp','sexn','q_obs','q_weight','q_lsweight','q_lshrsweight','p_obs','p_weight','p_lsweight']
    cdf1 = df[collapse_vars].groupby(['yearn','edun','pexp','sexn']).sum()
    cdf1 = cdf1.dropna()

    collapse_vars = ['yearn','edun','pexp','sexn','rwinc','winc_ws','rlnwinc','rhinc','rlnhinc','p_weight']
    fvars = ['winc_ws','rwinc','rlnwinc','rhinc','rlnhinc']
    colgroup = ['yearn','edun','pexp','sexn']
    cdf2 = df[collapse_vars].dropna()
    cdf2 = cdf2.groupby(colgroup).filter(lambda g: g['p_weight'].sum() > 0)
    cdf2 = cdf2.groupby(colgroup).apply(lambda x : pd.Series(np.average(x[fvars],weights=x['p_weight'],axis=0),fvars))
    cdf2 = cdf2.dropna()

    cdf=pd.merge(cdf2,cdf1,how='left',on=colgroup)
    cdf = cdf.reset_index()

    cdf.loc[:,'lnrwinc'] = np.log(cdf.rwinc)
    cdf.loc[:,'lnrhinc'] = np.log(cdf.rhinc)

    #print(df.yearn.value_counts())
    #print(df.groupby('yearn').mean()[['winc_ws','wgt','wgt_wks','wgt_hrs']])
    #print(df.groupby('yearn').mean()[['rwinc','q_obs','p_obs','ftfy']])
    #print(df.groupby('yearn').mean()[['tcwkwg','tchrwg','bcwkwg']])
    #print(df.groupby('yearn').mean()[['bchrwg','bchrwgkm','bcwkwgkm']])
    #print(cdf1.groupby('yearn').sum()[['q_obs','p_obs','q_lsweight','p_lsweight']])
    #print(cdf.groupby('yearn').sum()[['rwinc','winc_ws']])


    return df,cdf


def LaborSupplyFinal(df,autor=False):
    """
    Intput: data with binary college/no-college classification
    Output: Population in each geographic location by binary college/no college
    Categories here include the variables:
    """

    ### experience cats
    df.loc[(df.pexp >=0) & (df.pexp <=9),'expcat'] = 'exp0_9'
    df.loc[(df.pexp >=10) & (df.pexp <=19),'expcat'] = 'exp10_19'
    df.loc[(df.pexp >=20) & (df.pexp <=29),'expcat'] = 'exp20_29'
    df.loc[(df.pexp >=30) & (df.pexp <=39),'expcat'] = 'exp30_39'
    df.loc[(df.pexp >=40),'expcat'] = 'exp40_p'
    pexp_cats = ['exp0_9','exp10_19','exp20_29','exp30_39','exp40_p']
    pexp_type = CategoricalDtype(categories=pexp_cats, ordered=True)
    df.expcat = df.expcat.astype(pexp_type)

    ### experience cats
    """
    if using Autor's education cats, still need to multiply the share of csm by 0.5
    So we make it part of college here but only half of its weight will be part of college
    """
    if autor:
        df.loc[df.edun.isin(['hsc','lhs']),'college'] = 'nocol'
        df.loc[df.edun.isin(['clg','gtc','csm']),'college'] = 'col'
    else:
        df.loc[df.edun.isin(['hsc','lhs']),'college'] = 'nocol'
        df.loc[df.edun.isin(['clg','gtc','csm']),'college'] = 'col'

    ### make male, highschool grad, with 10 yers of experience as the reference wage for each year
    s = df[(df.edun=='hsc') & (df.pexp==10) & (df.sexn=='male')][['yearn','rwinc']]
    s.columns = ['yearn','refwage']
    df = pd.merge(df,s,how='left', on='yearn')
    df.loc[:,'relwage'] = df.rwinc/df.refwage

    """
    Calculate the average wage for each cell in our categories
    a cell mean, is the mean wage for a the average american by
    (this average is across all years  --  it's been adjusted for inflation:
        1. education
        2. gender
        3. years of experience in the job
    """
    s = df.groupby(['edun','sexn','pexp'])[['relwage']].mean()
    s.columns = ['celleu']
    s.reset_index()
    df = pd.merge(df,s,how='left', on=['edun','sexn','pexp'])

    #print(df.groupby('yearn').mean()[['relwage','refwage','celleu']])
    df.loc[:,'cellwgs'] = df.celleu * df.q_lshrsweight

    """
    #######################################
    ############ across years
    ########################################
    """
    """
    Create a set of total weights by year
    """
    ## no sex disagregation
    s = df.groupby(['yearn'])[['cellwgs']].sum()
    s.columns = ['tot_euwt']
    s.reset_index()

    ## disaggregate by sex
    s2 = df.groupby(['yearn','sexn'])[['cellwgs']].sum()
    s2 = s2.unstack('sexn')
    s2.columns = ['tot_euwt_f','tot_euwt_m']
    tmp = pd.concat([s,s2],axis=1)
    tmp.reset_index()
    f = pd.merge(df,tmp,how='left',on='yearn')

    """
    Create a set of weight shares based on the categories of total weights
    """
    f.loc[:,'sh_euwt'] = f.cellwgs / f.tot_euwt
    f.loc[f.sexn=='female','sh_euwt_f'] = f.cellwgs / f.tot_euwt_f
    f.loc[f.sexn=='male','sh_euwt_f'] = 0
    f.loc[f.sexn=='male','sh_euwt_m'] = f.cellwgs / f.tot_euwt_m
    f.loc[f.sexn=='female','sh_euwt_m'] = 0
    print(f.groupby('yearn').mean()[['sh_euwt','sh_euwt_f','sh_euwt_m']])
    """
    Finally, create shares
    """

    b_crit = f.edun.isin(['clg','gtc'])
    f_crit = f.edun.isin(['clg','gtc']) & (f.sexn=='female')
    m_crit = f.edun.isin(['clg','gtc']) & (f.sexn=='male')
    b_sc_crit = (f.edun=='csm')
    f_sc_crit = (f.edun=='csm') & (f.sexn=='female')
    m_sc_crit = (f.edun=='csm') & (f.sexn=='male')
    
    s = f[b_crit].groupby(['yearn'])[['sh_euwt']].sum() + 0.5*f[b_sc_crit].groupby(['yearn'])[['sh_euwt']].sum()
    s.columns = ['eu_shclg']
    s1 = f[f_crit].groupby(['yearn'])[['sh_euwt_f']].sum() + 0.5*f[f_sc_crit].groupby(['yearn'])[['sh_euwt_f']].sum()
    s1.columns = ['eu_shclg_f']
    s2 = f[m_crit].groupby(['yearn'])[['sh_euwt_m']].sum() + 0.5*f[m_sc_crit].groupby(['yearn'])[['sh_euwt_m']].sum()
    s2.columns = ['eu_shclg_m']

    tot1 = pd.concat([s,s1,s2],axis=1)
    tot1 = tot1.reset_index()

    print(tot1)
    tot1.loc[:,'eu_lnclg'] = np.log(tot1.eu_shclg/(1-tot1.eu_shclg))
    tot1.loc[:,'eu_lnclg_f'] = np.log(tot1.eu_shclg_f/(1-tot1.eu_shclg_f))
    tot1.loc[:,'eu_lnclg_m'] = np.log(tot1.eu_shclg_m/(1-tot1.eu_shclg_m))
    tot1.reset_index()

    #######################################
    ############ across years
    ########################################
    """
    Our final disaggregation is by skill and and sex so we add up
    the relative wages of all of these workers. Then we get a relative share
    """
    ## no sex disagregation
    s = df.groupby(['yearn','expcat'])[['cellwgs']].sum()
    s.columns = ['tot_euwt']
    s.reset_index()

    ## disaggregate by sex
    s2 = df.groupby(['yearn','expcat','sexn'])[['cellwgs']].sum()
    s2 = s2.unstack('sexn')
    s2.columns = ['tot_euwt_f','tot_euwt_m']
    tmp = pd.concat([s,s2],axis=1)
    tmp.reset_index()
    f = pd.merge(df,tmp,how='left',on=['yearn','expcat'])

    """
    Create a set of weight shares based on the categories of total weights
    """
    f.loc[:,'sh_euwt'] = f.cellwgs / f.tot_euwt
    f.loc[f.sexn=='female','sh_euwt_f'] = f.cellwgs / f.tot_euwt_f
    f.loc[f.sexn=='male','sh_euwt_f'] = 0
    f.loc[f.sexn=='male','sh_euwt_m'] = f.cellwgs / f.tot_euwt_m
    f.loc[f.sexn=='female','sh_euwt_m'] = 0

    """
    Finally, create shares
    """
    s = f[b_crit].groupby(['yearn','expcat'])[['sh_euwt']].sum() + 0.5*f[b_sc_crit].groupby(['yearn','expcat'])[['sh_euwt']].sum()
    s.columns = ['euexp_shclg']
    s1 = f[f_crit].groupby(['yearn','expcat'])[['sh_euwt_f']].sum() + 0.5*f[f_sc_crit].groupby(['yearn','expcat'])[['sh_euwt_f']].sum()
    s1.columns = ['euexp_shclg_f']
    s2 = f[m_crit].groupby(['yearn','expcat'])[['sh_euwt_m']].sum() + 0.5*f[m_sc_crit].groupby(['yearn','expcat'])[['sh_euwt_m']].sum()
    s2.columns = ['euexp_shclg_m']

    tot2 = pd.concat([s,s1,s2],axis=1)
    tot2 = tot2.reset_index()
    #f = pd.merge(f,tot2,how='left',on='yearn')
    tot2.loc[:,'euexp_lnclg'] = np.log(tot2.euexp_shclg/(1-tot2.euexp_shclg))
    tot2.loc[:,'euexp_lnclg_f'] = np.log(tot2.euexp_shclg_f/(1-tot2.euexp_shclg_f))
    tot2.loc[:,'euexp_lnclg_m'] = np.log(tot2.euexp_shclg_m/(1-tot2.euexp_shclg_m))
    d2 = tot2.yearn.count()
    tot = pd.merge(tot1,tot2,how='right',on='yearn')
    d = tot.yearn.count()
    assert d2 == d

    return tot,tot1,tot2


