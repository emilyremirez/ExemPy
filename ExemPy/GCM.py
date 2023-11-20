"""
Created on Sat Sep 03 2022
@author: Emily Remirez (eremirez@berkeley.edu)

"""

"""Functions for implementing the Generalized Context Model for speech perception."""

import math
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.optimize import minimize
import seaborn as sns

def activation(testset, cloud, dimsdict, c = 25):
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
    dims = dimsdict.copy()
    dims.update((x, (y/sum(dims.values()))) for x, y in dims.items())   # Normalize weights to sum to 1
    
    # If the testset happens to have N in it, remove it before joining dfs 
    test = testset.copy()
    if 'N' in test.columns:
        test = test.drop(columns='N', axis=1,inplace=True)
    
    exemplars = cloud.copy()

    # Merge test and exemplars
    bigdf = pd.merge(
        test.assign(key = 1),         # Add column named 'key' with all values == 1
        exemplars.assign(key = 1),    # Add column named 'key' with all values == 1
        on = 'key',                   # Match on 'key' to get cross join (cartesian product)
        suffixes = ['_t', '_ex']
    ).drop('key', axis=1)           # Drop 'key' column
    
    
    dimensions = list(dims.keys())                # Get dimensions from dictionary
    weights = list(dims.values())                 # Get weights from dictionary
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
    bigdf['a'] = np.exp(-bigdf.dist*c) * bigdf.N
    return bigdf


def exclude(cloud, test, exclude_self = True, alsoexclude = None): 
    '''
    Removes specific rows from the cloud of exemplars, to be used
    prior to calculating activation. Prevents activation from being
    overpowered by stimuli that are too similar to particular exemplars.
    E.g., prevents comparison of a stimulus to itself, or to exemplars
    from same speaker. Returns dataframe containing a subset of
    rows from the cloud.
    
    Required parameters:
    
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar
    
    test = single row dataframe containing the stimulus to be categorized
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
    
    Optional parameters:
    
    alsoexclude = a list of strings matching columns in the cloud
        categories) to exclude if value is the same as that of the test.
        (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    '''
    # Make a copy of the cloud and call it exemplars. 
    #    This is what we'll return at the end
    exemplars = cloud.copy()
    
    
    # Remove the stimulus from the cloud
    if exclude_self == True:
        exemplars = exemplars[~exemplars.isin(test)].dropna()  
    
    if alsoexclude != None:
        if type(alsoexclude) != list:
            alsoexclude = [alsoexclude]
        for feature in alsoexclude:
            featval = test[feature].iloc[0]
            exclude_exemps = exemplars[exemplars[feature] == featval].index
            exemplars = exemplars.drop(exclude_exemps)
    
    return exemplars


def reset_N(exemplars, N = 1):
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
    bias_N will set N values according to values in a dictionary.
    That is, within a category type, each category will have the N
    value specified in the dictionary
    
    Required parameters:
    
    exemplars = dataframe of exemplars to which the stimulus is being compared
    
    cat = a string designating the category type which is being primed
    
    catbias = dictionary with categories (e.g. vowels) as keys and N value for the  
        category as values
    '''
    extemp = exemplars.copy()
    extemp['N'] = extemp[cat].map(catbias)
    return extemp


def probs(bigdf, cats):    
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
            label = cat + '_ex'
            
        # Sum up activation for every label within that category
        cat_a = bigdf.groupby(label).a.sum()
        # Divide the activation for each label by the total activation for that category
        pr = cat_a / sum(cat_a)
        # rename a for activation to probability
        pr = pr.rename_axis(cat).reset_index().rename(columns={"a" : "probability"})
        # add this to the dictionary 
        prs[cat] = pr
    return prs


def choose(probsdict, test, cats, fc = None):
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

        
    fc = Dict where keys are category names in the dataframe and
        values are a list of category labels.
        Used to simulate a forced choice experiment
        in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']},
        the choice will be the alternative 
        with higher probability, regardless of whether other vowels not
        listed have higher probabilities. 
        There can be any number of alternatives in the list.
    
    '''
    newtest = test.copy()      # make a copy of the test set to add to
    pr = probsdict.copy()        # make a copy of the probs dict to subset if forced choice is set
    choice = ''
    choiceprob = 1
    
    # If using forced choice, restrict the choices to the terms 
    # This doesn't change the probability! So something could have a low prob,
    ## but still be the winner
    if fc != None: 
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
        prob = dframe['probability']
        winner = dframe.loc[prob==max(prob)]
            
        # if more than one winner, choose randomly
        if len(winner) > 1:
            winner = winner.sample(1)
                                                 
        choice = winner[cat].item()
        choiceprob = winner['probability'].item()
        
        newtest[choicename] = choice
        newtest[choiceprobname] = choiceprob      
    return newtest

def wideprobs(cats,pr):
    '''
    Get a list of probabilities reshaped to a wide format.
    
    Required parameters:
    
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    
    pr = Output of probs function. Dictionary of dictionaries.
        Each key is a category; values are dictionaries where keys are
        labels and values represent probability of the stimulus
        being categorized into that label.
    '''
    widelist=[]
    for cat in cats:
        pref = str(cat+"_")       
        dframe = pr[cat]
        wide = dframe.set_index(cat).transpose().reset_index(drop=True).add_prefix(pref)
        widelist.append(wide)
    return widelist

def probsdf(widelist, test):
    '''
    Alternative to the choose function. Rather than picking the category labels
    with the highest probability, join the wide format probalities alongside the stimulus.
    Returns the larger dataframe.
    
    Required parameters:
    
    widelist = Output of the wideprobs function. A list of probabilities for each category
        label, reshaped to wide format.
    
    test = single line data frame representing the test/stimulus being categorized
    '''
    newdf = test
    for wide in widelist:    
        newdf = pd.merge(
            newdf.assign(key = 1),
            wide.assign(key = 1),
            on = 'key').drop('key', axis=1)
    return newdf


def categorize(testset, cloud, cats, dimsdict, c, 
               exclude_self = True, alsoexclude = None, N=1, fc=None):
    '''
    Categorizes a stimulus based on functions defined in library. 
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with
    added columns showing what was 
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
    
    dimsdict = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to .01. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    alsoexclude = a list of strings matching columns in the cloud (categories)
        to exclude  if value is the same as that of the test.
        (E.g., to exclude all exemplars from the speaker
        to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        


    '''
    exemplars = cloud.copy()
    test = testset
    exemplars = exclude(exemplars, test, exclude_self = exclude_self, alsoexclude = alsoexclude)
    exemplars = reset_N(exemplars, N = N)
    bigdf = activation(test, exemplars, dimsdict = dimsdict, c = c)
    pr = probs(bigdf, cats)
    choices = choose(pr, test, cats, fc = fc)
    return choices 


def multicat(testset, cloud, cats, dimsdict, c = 25, N = 1, biascat = None, catbias = None,
                 exclude_self = True, alsoexclude = None,  fc = None):
    '''
    Categorizes a dataframe of 1 or more stimuli based on functions defined in library
    
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with added columns
    showing what was chosen for a category and with what probability
    
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
        
    dimsdict = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to 25. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    
    biascat = A string indicating the category type to be biased
        or primed on (e.g. 'vowel', 'speaker')
    
    catbias = Dict where keys are categories of biascat and values are
        ints that indicate relative N values. (e.g., {'i':5,'a':1} would make every 'i' exemplar 
        contribute 5 times as much activation as each 'a)
    
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
    
        
    fc = Dict where keys are category names in the dataframe and values are a list of category labels.
        Used to simulate a forced choice experiment in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']}, the choice will be the alternative 
        with higher probability, regardless of whether other vowels not listed have higher probabilities. 
        There can be any number of alternatives in the list. 

     
    '''
    choicelist=[]
    prlist=[]
    for ix in list(testset.index.values):
        # Reload exemplars within the loop
        ## if not, exemplars shrinks every time you use exclude()!
        exemplars = cloud.copy()   
        test = testset.loc[[ix,]]
        
        # exclusions
        exemplars = exclude(exemplars, test, exclude_self = exclude_self, alsoexclude = alsoexclude)
        
        #add N 
        if catbias != None: 
            exemplars = bias_N(exemplars, biascat, catbias)
        else: exemplars = reset_N(exemplars, N = N)
        
        # calculate probabilities
        bigdf = activation(test, exemplars, dimsdict = dimsdict, c = c)
        pr = probs(bigdf, cats)
        
        # Luce's choice rule
        choicerow = choose(pr, test, cats, fc = fc)       
        choicelist.append(choicerow)
        # Get probabilities 
        widelist = wideprobs(cats, pr)
        widerow = probsdf(widelist,test)
        widelist.append(widerow)
    choices = pd.concat(choicelist, ignore_index = True)
 
    return choices


