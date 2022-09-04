"""
Created on Wed January 20, 2021
Last updated July 21, 2021
@author: Emily Remirez (eremirez@berkeley.edu)
@version: 0.1
"""

import math
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.optimize import minimize
import seaborn as sns
sns.set(style='ticks', context='paper')
colors=["#e3c934","#68c4bf","#c51000","#287271"]
sns.set_palette(colors)

def HzToBark(cloud,formants):
    '''
    Convert selected columns from Hz to Bark scale. Renames the formants as z.
    Returns the data frame with additional columns: the value of the formant
    converted from Hz to Bark
    
    Required parameters:
    
    cloud = dataframe of exemplars 
    
    formants = list of formants to be converted 
    '''
    # Make a copy of the cloud
    newcloud = cloud.copy()
    
    # For each formant listed, make a copy of the column prefixed with z
    for formant in formants:
        for ch in formant:
            if ch.isnumeric():
                num=ch
        formantchar = (formant.split(num)[0])
        name = str(formant).replace(formantchar,'z')
        # Convert each value from Hz to Bark
        newcloud[name] = 26.81/ (1+ 1960/newcloud[formant]) - 0.53
    # Return the dataframe with the changes
    
    return newcloud


def activation(testset,cloud,dims,c=25):
    '''
    Calculate activation for all exemplars stored in the cloud
    with respect to some stimulus, referred to as test. Returns
    a data frame with column 'a' added for each row.
    
    Required parameters:
    
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
    
    dimsdict = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to 25. 
        
    '''
    # Get stuff ready
    dims.update({key : abs(val) for key, val in dims.items()})
    dims.update((x, (y/sum(dims.values()))) for x, y in dims.items())   # Normalize weights to sum to 1
    
    # If the testset happens to have N in it, remove it before joining dfs 
    test=testset.copy()
    if 'N' in test.columns:
        test = test.drop(columns='N', axis=1,inplace=True)
    
    exemplars=cloud.copy()

    # Merge test and exemplars
    bigdf = pd.merge(
        test.assign(key=1),         # Add column named 'key' with all values == 1
        exemplars.assign(key=1),    # Add column named 'key' with all values == 1
        on='key',                   # Match on 'key' to get cross join (cartesian product)
        suffixes=['_t', '_ex']
    ).drop('key', axis=1)           # Drop 'key' column
    
    
    dimensions=list(dims.keys())                # Get dimensions from dictionary
    weights=list(dims.values())                 # Get weights from dictionary
    tcols = [f'{d}_t' for d in dimensions]      # Get names of all test columns
    excols = [f'{d}_ex' for d in dimensions]    # Get names of all exemplar columns
    
    
    # Multiply each dimension by weights
    i = bigdf.loc[:, tcols].values.astype(float)     # Get all the test columns
    i *= weights                                     # Multiply test columns by weight
    j = bigdf.loc[:, excols].values.astype(float)    # Get all the exemplar columns
    j *= weights                                     # Multiply exemplar columns by weights
    
    # Get Euclidean distance
    bigdf['dist'] = np.sqrt(np.sum((i-j)**2, axis=1))
    
    # get activation: exponent of negative distance * sensitivity c, multiplied by N_j
    bigdf['a'] = np.exp(-bigdf.dist*c)*bigdf.N
    
    return bigdf
    
    
def exclude(cloud, test, exclude_self=True, alsoexclude=None): 
    '''
    Removes specific rows from the cloud of exemplars, to be used
    prior to calculating activation. Prevents activation from being
    overpowered by stimuli that are too similar to particular exemplars.
    E.g., prevents comparison of a stimulus to itself, or to exemplars from same speaker
    Returns dataframe containing a subset of rows from the cloud.
    
    Required parameters:
    
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar
    
    test = single row dataframe containing the stimulus to be categorized
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
    
    Optional parameters:
    
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    
    '''
    # Make a copy of the cloud and call it exemplars. 
    #    This is what we'll return at the end
    exemplars = cloud.copy()
    
    # Remove the stimulus from the cloud
    if exclude_self == True:
        exemplars=cloud[~cloud.isin(test)].dropna()  
    
    if alsoexclude != None:
        for feature in alsoexclude:
            featval=test[feature].iloc[0]
            exclude_exemps=exemplars[ exemplars[feature] == featval ].index
            exemplars.drop(exclude_exemps, inplace=True)
        
    return exemplars


def reset_N(exemplars, N=1):      # Add or override N, default to 1
    '''
    Adds an N (base activation) column to the exemplar cloud so
    that activation with respect to the stimulus can be calculated
    Default value is 1, i.e., equal activation for each exemplar.
    Returns the exemplar data frame with added or reset column
    
    Required parameters:
    
    exemplars = data frame of exemplars to which the stimulus is being
        compared
        
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
    '''
    extemp = exemplars.copy()
    extemp['N'] = N
    
    return extemp


def bias_N(exemplars, cat, catbias):
    '''
    Adds or overwrites an N (base activation) colummn to the exemplar 
    cloud so that activation with respect to the stimulus can be 
    calculated. Unlike reset_N, which assigns the same N value to all exemplars,
    bias_N will set N values according to values in a dictionary. That is, within a 
    category type, each category will have the N value specified in the dictionary
    
    Required parameters:
    
    exemplars = dataframe of exemplars to which the stimulus is being compared
    
    cat = a string designating the category type which is being primed
    
    catbias = dictionary with categories (e.g. vowels) as keys and N value for the  
        category as values
    '''
 
    extemp = exemplars.copy()
    extemp['N'] = extemp[cat].map(catbias)
    
    return extemp


def probs(bigdf,cats):
    
    '''
    Calculates the probability that the stimulus will be categorized with a
    particular label for a given category (e.g., vowel labels 'i', 'a', 'u' for
    the category 'vowel'). Probability is calculated by summing the activation
    across all exemplars sharing a label, and dividing that by the total amount
    of activation in the system for the category. Returns a dictionary of dictionaries.
    Each key is a category; values are dictionaries where keys are labels and values
    represent probability of the stimulus being categorized into that label.
    
    Required parameters: 
    
    bigdf = a dataframe produced by activation(), which contains a row for each
        exemplar with the additional column 'a' representing the amount of 
        activation for that exemplar with respect to the stimulus
    
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    prs = {}
    
    if type(cats) != list:
        cats = [cats]
    
    # Loop over every category in the list of categories
    for cat in cats: 
        if cat in bigdf:
            label = cat
        else: 
            # make category match the exemplar category in name if i and j share column names
            label = cat+'_ex'
            
        # Sum up activation for every label within that category
        cat_a = bigdf.groupby(label).a.sum()
        # Divide the activation for each label by the total activation for that category
        pr = cat_a/sum(cat_a)
        # rename a for activation to probability
        pr = pr.rename_axis(cat).reset_index().rename(columns={"a":"probability"})
        # add this to the dictionary 
        prs[cat]=pr
    
    return prs


def choose(probsdict,test,cats,runnerup=False,fc=None):
    '''
    Chooses a label for each category which the stimulus will be categorized as.
    Returns the test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability. Optionally will give the
    second most probable label as well. 
    
    Required parameters:
    pr = dictionary of probabilities, given from probs(). Each key should represent
        a category (e.g. 'vowel'), with values as dataframe. Dataframe should
        have a probability for each category label
        
    test = single line data frame representing the test/stimulus being categorized
    
    cats = list of categories to be considered (e.g., ["vowel"])
            
    Optional parameters:
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False. 
        
    fc = Dict where keys are category names in the dataframe and values are a list of category labels.
        Used to simulate a forced choice experiment in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']}, the choice will be the alternative 
        with higher probability, regardless of whether other vowels not listed have higher probabilities. 
        There can be any number of alternatives in the list.
    
    '''
    newtest = test.copy()      # make a copy of the test set to add to
    pr=probsdict.copy()        # make a copy of the probs dict to subset if forced choice is set
    choice = ''
    choiceprob = 1
    
    
    if fc!=None: 
        fccats = fc.keys()
        for fccat in fccats:
            options = fc[fccat]
            scope = probsdict[fccat]
            toconsider = scope.loc[scope[fccat].isin(options)]
        pr[fccat] = toconsider

    for cat in cats:
        choicename = cat + 'Choice'
        choiceprobname = cat + 'Prob'
        
        dframe = pr[cat]
        prob=dframe['probability']
        
        winner = dframe.loc[prob==max(prob)]
                                                 
        choice = winner[cat].item()
        choiceprob = winner['probability'].item()
        
        newtest[choicename] = choice
        newtest[choiceprobname] = choiceprob

    return newtest


def gettestset(cloud,balcat,n):     #Gets n number of rows per cat in given cattype
    '''
    Gets a random test set of stimuli to be categorized balanced across a particular
    category, e.g., 5 instances of each label 'i','a', 'u' for category 'vowel'. 
    Returns a data frame of stimuli.
    
    Required parameters:
    
    cloud = dataframe of exemplars
        
    balcat = category stimuli should be balanced across 
        
    n = number of stimuli per category label to be included
    '''
    testlist=[]
    for cat in list(cloud[balcat].unique()):
        samp = cloud[cloud[balcat]==cat].sample(n)
        testlist.append(samp)
    test=pd.concat(testlist)
    
    return test


def categorize(testset,cloud,cats,dims,c,exclude_self=True,alsoexclude=None, N=1, runnerup=False, fc=None):
    '''
    Categorizes a stimulus based on functions defined in library. 
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability
    
    Required parameters:
    
    testset = a dataframe with one row, a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
        
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    
    dims = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to .01. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False.

    '''
    
    test=testset
    exemplars=exclude(cloud,test,exclude_self=exclude_self,alsoexclude=alsoexclude)
    reset_N(exemplars, N=N)
    bigdf=activation(test,exemplars,dims=dims,c=c)
    pr=probs(bigdf,cats)
    choices=choose(pr,test,cats,runnerup=runnerup,fc=fc)
    
    return choices 


def getactiv(activation,x,y,cat):
    
    """ 
    Creates a simplified data frame showing the activation for each exemplar 
    with respect to the stimulus. Primarily for use with the activplot()
    function. 
    
    Required parameters:
    
    activation = DataFrame resulting from the activation() function, containing
        one row per stored exemplar, with their activation 'a' as a column
        
    x = String. Dimension to be plotted as x axis in scatterplot (e.g., F2). Matches
        the name of a column in the activation DataFrame.
    
    y = String. Dimension to be plotted as y axis in scatterplot (e.g., F1). Matches
        the name of a column in the activation DataFrame.
    
    cat = String. Category used to color code exemplars in scatter plot. Matches the name
        of a column in the activation DataFrame.
    """
    renamedict={}
    serieslist=[activation['a']]

    for item in (x,y,cat):
        name = str(item+"_ex")
        if name not in activation:
            name = item
        renamedict[name] = item
        activseries =activation[name]
        serieslist.append(activseries)
        
    activ=pd.concat(serieslist,axis=1)
    activ.rename(columns=renamedict,inplace=True)
    
    return activ


def activplot(a,x,y,cat, test, invert =True):
    """
    Plots each exemplar in x,y space according to specified dimensions. Labels within
    the category are grouped by color. The stimulus or test exemplar is plotted in dark
    blue on top of exemplars. Note: axes are inverted, assuming F1/F2 space
    
    Required parameters:
    
    a = DataFrame produced by getactiv() function. Contains a row for each exemplar
    
    x = String. Dimension to be plotted as x axis in scatterplot (e.g., F2). Matches
        the name of a column in the activation DataFrame.
    
    y = String. Dimension to be plotted as y axis in scatterplot (e.g., F1). Matches
        the name of a column in the activation DataFrame.
    
    cat = String. Category used to color code exemplars in scatter plot. Matches the name
        of a column in the activation DataFrame.
    
    test = name of test exemplar, one row of a DataFrame.
    
    invert = Boolean. Specifies whether axes should be inverted (as for a vowel space). Defaults to true.
        
    """
    
    pl = sns.scatterplot(data=a,x=x,y=y,hue=cat,size='a',size_norm=(0,a.a.max()),
                     alpha=0.5,sizes=(5,100),legend=False)
    pl = sns.scatterplot(data=test, x=x,y=y,alpha=.5,color='darkblue',marker="X", s= 50, legend=False)
    
    if invert == True:
        pl.invert_xaxis()
        pl.invert_yaxis()

    return pl


    def multicat(testset,cloud,cats,dims,c=25,N=1,biascat=None,catbias=None,rescat=None, ncyc= None,
                 exclude_self=True,alsoexclude=None,runnerup=False,fc=None):
    '''
    Categorizes a dataframe of 1 or more stimuli based on functions defined in library
    
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability
    
    Required parameters:
    
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
        
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
        
    dims = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to 25. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    
    biascat = A string indicating the category type to be biased or primed on (e.g. 'vowel', 'speaker')
    
    catbias = Dict where keys are categories of biascat and values are
        ints that indicate relative N values. (e.g., {'i':5,'a':1} would make every 'i' exemplar 
        contribute 5 times as much activation as each 'a)
    
    rescat = Category to resonate on. If given, 
    
    ncyc = Int indicating how many cycles of resonance
    
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False.
        
    fc = Dict where keys are category names in the dataframe and values are a list of category labels.
        Used to simulate a forced choice experiment in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']}, the choice will be the alternative 
        with higher probability, regardless of whether other vowels not listed have higher probabilities. 
        There can be any number of alternatives in the list. 
    '''
    
    choicelist=[]
    for ix in list(testset.index.values):
        test = testset.loc[[ix,]]
        
        # exclusions
        exemplars=exclude(cloud,test,exclude_self=exclude_self,alsoexclude=alsoexclude)
        
        #add N 
        if catbias != None: 
            exemplars = bias_N(exemplars,biascat,catbias)
        else: exemplars = reset_N(exemplars, N=N)
        
        # calculate probabilities
        bigdf=activation(test,exemplars,dims = dims,c=c)
        pr=probs(bigdf,cats)
        
        # resonate if applicable -- recalculate probs based on a resonance term
        if rescat != None:
            for n in range(0,ncyc):
                edict = pr[rescat].set_index(rescat).to_dict()['probability']
                # resonance term = probability of category divided by number of cycles
                    ## so that effect decays over time
                exemplars['resterm'] = exemplars[rescat].map(edict) / (n+1)
                # Add resterm to N value; N only ever goes up
                exemplars['N'] = exemplars['N'] + exemplars['resterm']
                bigdf=activation(test,exemplars,dims = dims,c=c)
                pr=probs(bigdf,cats)
        
        # Luce's choice rule
        choices = choose(pr,test,cats,runnerup=runnerup,fc=fc)
        
        choicelist.append(choices)
    choices=pd.concat(choicelist, ignore_index=True)
    
    return choices


def checkaccuracy(choices,cats):
    '''
    Check rather the choices made by the model match the 'intended' label for each category.
    Returns a copy of the testset dataframe with column added indicating whether the choice for
    each category was correct (y) or incorrect (n)
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        category's probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    if type(cats) != list:
        cats = [cats]
    
    acc = choices.copy()                     # Make a copy of choices to muck around with
    
    for cat in cats:                     # Iterate over your list of cats
        accname = cat + 'Acc'            # Get the right column names
        choicename = cat + 'Choice'
        
        # If choice is the same as intended, acc =y, else n
        acc[accname] = np.where(acc[cat]==acc[choicename], 'y', 'n')      
    
    return acc

def propcorr(acc,cat):
    '''
    Calculates the proportion of stimuli under each label which were categorized correctly
    Returns a dataframe with keys as labels and values as proportions between 0 and 1.
    
    Required parameters:
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    '''
    perc = dict(acc.groupby(cat)[cat+'Acc'].value_counts(normalize=True).drop(labels='n',level=1).reset_index(level=1,drop=True))
    pc=pd.DataFrame.from_dict(perc, orient='index').reset_index()
    pc.columns=[cat,'propcorr']
    
    return pc


def overallacc(acc,cat):
    '''
    Calculates accuracy for categorization overall, across all labels. Returns a 
    proportion between 0 and 1. 
    
    Required parameters: 
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    '''
    
    totalcorrect = acc[cat+'Acc'].value_counts(normalize=True)['y']
    
    return totalcorrect


def accplot(acc,cat):
    '''
    Plots a bar graph showing the proportion of trials which were categorized
    veridically, that is, accuracy of categorization.
    
    Required parameters:
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    
    '''
    perc = dict(acc.groupby(cat)[cat+'Acc'].value_counts(normalize=True).drop(labels='n',level=1).reset_index(level=1,drop=True))
    pc=pd.DataFrame.from_dict(perc, orient='index').reset_index()
    pc.columns=[cat,'propcorr']
    
    obs=str(len(acc))
    pl = sns.barplot(x=cat,y='propcorr',data=pc,palette=colors)
    plt.ylim(0,1.01)
    pl.set(ylabel='Proportion accurate of '+obs+' trials')
    pl.set_xticklabels(
    pl.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
    plt.show()
    
    return pl


def confusion(choices,cats):
    '''
    Returns a confusion matrix comparing intended category with categorization.
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        categories probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    if type(cats) != list:
        cats = [cats]
    
    matrices={}
    for cat in cats:
        matrices[cat]=pd.crosstab(choices[cat],choices[cat+'Choice'], normalize='index').round(2).rename_axis(None)
    
    return matrices


def errorfunc(x, testset, cloud, dimslist, cat):
    ''' 
    Returns a proportion representing the total amount of error for a single category that
    the categorizer makes given a certain set of c and w values. This is intended to
    be used with an optimization function so that the total amount of error can be 
    minimized; that is, the accuracy can be maximized. 
    Note that z0 is automatically set to 1.
    
    Required parameters: 
    
    x = a vector of values to be used by multicat. x[0] should be c, x[1], x[2], x[3]
        should correspond to dimslist[1], dimslist[2], dimslist[3]
        
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the dims list. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims list
    
    dimslist = a list of dimensions (e.g., formants), for which weights w should be given,
        and along which exemplars should be compared.
    
    cat = the category, 
        
    
    '''
    #x = [c,z1,z2,z3]
    catlist=[cat]
    c=x[0]
    dimsdict={dimslist[0]:1,dimslist[1]:x[1],dimslist[2]:x[2],dimslist[3]:x[3]}
    choices=multicat(cloud,testset,catlist,dims=dimsdict,c=c)
    accuracy=checkaccuracy(choices,catlist)
    err = accuracy[cat+'Acc'].value_counts(normalize=True)['n']
    
    return err


def continuum (data, start, end, dimslist, steps=7, stimdetails=False):
    '''
    Returns a continuum dataframe with interpolated values
    from a start to end value with a given number of steps
    * Users should be sure to specify any and all parameters they want
    start and end to match for. That is, say there are 2 repetitions of
    a stimulus. If it doesn't matter whether start and end are from the same
    repetition, you do not need to specify repetition number; one row will
    be chosen randomly. If it *does* matter that they're the same repetition,
    be sure to include repetition number in the dictionary.
    
    Required parameters:
    
    data = DataFrame to draw start and end stimuli from
    
    start = Dictionary indicating properties of the desired start
        with category types as keys, and their desired category as values.
        e.g., {"vowel":"i","speaker"="LB"}
    
    end = Dictionary indicating properties of the desired start
        with category types as keys, and their desired category as values
        
    dimslist = list containing the names of dimensions to be interpolated
    
    Optional parameters: 
    
    steps = integer indicating the total number of continuum steps. Defaults to 7.
    
    stimdetails = Boolean, defaults to False. Debugging/auditing tool to
        get details of the stimulus that aren't preserved in the returned
        dataframe (e.g., speaker ID)
    '''
    # create a copy of the entire df to subset according to conditions
    # match category to value from dictionary, subset
    # repeat subsetting until all conditions are satisfied
    st=data.copy()
    for i in range(0,len(start)):
        cat = list(start.keys())[i]
        val = list(start.values())[i]
        condition = st[cat]==val
        st = st.loc[condition]
    # reset index has to be outside of the loop to work with >2 conditions
    # sample(1) is there to just pick an observation if the conditions don't point
    ## a unique row in the dataframe
    st = st.sample(1).reset_index()
    
    en=data.copy()
    for i in range(0,len(end)):
        cat = list(end.keys())[i]
        val = list(end.values())[i]
        condition = en[cat]==val
        en = pd.DataFrame(en.loc[condition])
    en = en.sample(1).reset_index()
    
    # remember start & end values if needed
    if stimdetails == True:
        print("Start: " , st.iloc[0])
        print("End: " , en.iloc[0])
    
    norms = {}
    for dim in dimslist:                      # Calculate the difference between start and end for each dim
        norms[dim] = en[dim] - st[dim] 
   
    vals={}
    rowlist = []
    for i in range (0,steps):
        for dim in dimslist: 
            vals[dim] = st[dim] + (norms[dim] * i/(steps-1))    # the values for each dim = start val + diff by step
            row = pd.DataFrame(vals)
        rowlist.append(row)

    contdf = pd.concat(rowlist,ignore_index=True)
    
    return contdf


def datasummary(dataset, catslist, dimslist):
    '''
    Creates dataframe of mean values grouped by catgories
    
    Required parameters: 
    
    dataset = A dataframe to be analyzed, where each row is an observation
        Requires at least one category and one dimension
        
    catslist = List of categories to group by. Also accepts string.
    
    dimslist = List of dimensions to get values for. Also accepts dict
        with dimensions as keys.
    '''
    # Convert cat to list (e.g. if only one term is given)
    if type(catslist) != list:
        catslist = [catslist]
    # If the weights dictionary is given instead of the dimlist,
    ## take just the keys as a list
    if type(dimslist) == dict:
        dimslist=list(dimslist.keys())
    
    # group by categories: cats[0] will be used to group first, then cats[1]
    # i.e., if cats = ["vowel","type"], vowel1-type1, vowel1-type2, vowel2-type1, vowel2-type2...
    # get the mean of values for each dimension grouped by categories
    df = dataset.groupby(catslist,as_index=False)[dimslist].mean()
    
    return df


def cpplot(datalist,cat,datanames=None, plot50=True):
    '''
    Generates a (cp = categorical perception) plot. On the X axis is the stimulus number,
    on the Y axis is the proportion of [label] responses with [label] being the label that
    was assigned to the first stimulus. Designed to be used with stimuli continua
    
    Required parameters:
    
    datalist = Designed to be output of multicat() or multicatprime(). Dataframe or list of dataframes
        containing each stimulus, what it was categorized as, and the probability
    
    cat = Type of category decision to visualize, e.g., 'vowel'
    
    Optional parameters:
    
    datanames = List of labels to use for each curve in the plot. Names should be in same
        order as in datalist
        
    plot50 = Boolean indicating whether a dashed line is added at 0.5 to aid in assessing
        boundaries in categorical perception. Defaults to true. 
    '''
    # Set up some labels
    if type(datalist) != list:
        datalist = [datalist]
    choicename = cat+'Choice'
    probname = cat+'Prob'
    # Get the label of the first stimulus
    stv = datalist[0].loc[0][choicename]
    
    def copy(d):
        d = d
        return d

    def inv(d):
        d = 1-d
        return d
    
    # get the inverse of probability if not first value, for each dataset
    curvelist = []
    i = 1
    j = 0
    for dataset in datalist:
        if datanames != None:
            lab = datanames[j]
        else:
            lab = "Data " + str(i)
        dataset['yax'] = dataset.apply(lambda x: copy(x[probname]) if x[choicename]==stv else inv(x[probname]),axis=1)
        curve = sns.lineplot(x=(dataset.index.values)+1, y="yax", data=dataset, label=lab)
        i += 1
        j += 1
        dataset.drop('yax', axis=1, inplace=True)

    # use the last dataset/plot to set axes and stuff
    p = curve
    # Add labels & plot
    yaxisname = "Proportion " + stv + " Response"
    p.set_ylabel(yaxisname)
    p.set_xlabel("Step")
    
    if plot50 == True:
        plt.axhline(y=0.5, color='gray', linestyle=':')
    
    plt.show()

