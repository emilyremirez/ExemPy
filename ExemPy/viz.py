"""
Created on Sat Sep 03 2022
@author: Emily Remirez (eremirez@berkeley.edu)

"""

"""Functions for visualizing ExemPy simulations."""

import math
import random
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.optimize import minimize
import seaborn as sns

def getactiv(activxn, x, y, cat):
    
    """ 
    Creates a simplified data frame showing the activation for each exemplar 
    with respect to the stimulus. Primarily for use with the activplot()
    function. 
    
    Required parameters:
    
    actxn = DataFrame resulting from the activation() function, containing
        one row per stored exemplar, with their activation 'a' as a column
        
    x = String. Dimension to be plotted as x axis in scatterplot (e.g., F2). Matches
        the name of a column in the activation DataFrame.
    
    y = String. Dimension to be plotted as y axis in scatterplot (e.g., F1). Matches
        the name of a column in the activation DataFrame.
    
    cat = String. Category used to color code exemplars in scatter plot. Matches the name
        of a column in the activation DataFrame.
    """

    renamedict = {}    
    activseries={"a" : activxn['a']}

    for item in (x, y, cat):
        name = str(item + "_ex")
        if name not in activxn:
            name = item
        renamedict[name] = item
        activseries[item] = activxn[name]
    activ = pd.DataFrame.from_dict(activseries)
    return activ     



def activplot(a, x, y, cat, test, invert = True):
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
    
    pl = sns.scatterplot(data = a,
                         x = x,
                         y = y,
                         hue = cat,
                         size = 'a',
                         size_norm = (0, a.a.max()),
                         alpha = 0.5,
                         sizes = (5, 100),
                         legend = False)
    pl = sns.scatterplot(data = test,
                         x = x,
                         y = y,
                         alpha = .5,
                         color = 'darkblue',
                         marker = "X",
                         s = 50,
                         legend = False)
    
    
    if invert == True:
        pl.invert_xaxis()
        pl.invert_yaxis()
    return pl


def accplot(acc, cat):
    '''
    Plots a bar graph showing the proportion of trials which were categorized
    veridically, that is, accuracy of categorization.
    
    Required parameters:
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    
    '''
    perc = dict(
        acc.groupby(cat)[cat+'Acc']
            .value_counts(normalize = True)
            .drop(labels = 'n', level = 1)
            .reset_index(level = 1, drop = True))
    pc = pd.DataFrame.from_dict(perc, orient = 'index').reset_index()
    pc.columns = [cat,'propcorr']
    
    obs = str(len(acc))
    pl = sns.barplot(x = cat, y = 'propcorr', data = pc)
    plt.ylim(0,1.01)
    pl.set(ylabel = 'Proportion accurate of ' + obs + ' trials')
    pl.set_xticklabels(
        pl.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right',
        fontweight = 'light',
        fontsize = 'x-large')
    plt.show()
    return pl

def cpplot(datalist, cat, alts, xax, datanames = None, plot50 = True):
    '''
    Generates a (cp = categorical perception) plot. On the X axis is the stimulus number,
    on the Y axis is the proportion of [label] responses with [label] being the label that
    was assigned to the first stimulus. Designed to be used with stimuli continua
    
    Required parameters:
    
    datalist = Designed to be output of multicat() or multicatprime(). Dataframe or list of dataframes
        containing each stimulus, what it was categorized as, and the probability
    
    cat = Type of category decision to visualize, e.g., 'vowel'
    
    alts = List indicating 
    
    Optional parameters:
    
    xax = String giving the name of a column to use as the x-axis on the plot; for example, the step
        number along a continuum
    
    datanames = List of labels to use for each curve in the plot. Names should be in same
        order as in datalist
        
    plot50 = Boolean indicating whether a dashed line is added at 0.5 to aid in assessing
        boundaries in categorical perception. Defaults to true. 
    '''
    if type(datalist) != list:
        datalist = [datalist]
    choicename = cat + 'Choice'
    probname = cat + 'Prob'
    stv = alts[0]
    env = alts[1]

    def copy(d):
        d = d
        return d
    def inv(d):
        d = 1-d
        return d

    j = 0
    for dataset in datalist:
        if datanames != None:                                                       
            lab = datanames[j]
        else:
            lab = "Data " + str(i)
        ## get the inverse of probability if not first value, for each dataset
            # If neither start nor end, set value to NaN
        dataset['yax'] = dataset.apply(
            lambda x: copy(x[probname]) if x[choicename] == stv
            else (inv(x[probname]) if x[choicename] == env else float("NaN")),
            axis = 1)
        dataset["Data"] = lab
        j += 1

    datalist = pd.concat(datalist)
    curve = sns.pointplot(
            x = xax,
            y = "yax",
            data = datalist,
            hue = "Data")

    for line, row in datalist.iterrows():
        if math.isnan(row["yax"]):
            realchoice = str(row[choicename])
            realprob = str(np.round(row[probname], 2))
            stimulusnumber = str(row[xax])
            dataset = str(row["Data"])
            print("----- Hey! -----")
            print("Stimulus", stimulusnumber)
            print("in dataset", dataset)
            print("was categorized as", realchoice)
            print("with probability", realprob)

    # Add labels & plot
    yaxisname = "Proportion " + stv + " Response"
    curve.set_ylabel(yaxisname)
    curve.set_xlabel("Step")
    curve.set_ylim(-0.05, 1.05)

    if plot50 == True:
        plt.axhline(y = 0.5, color = 'gray', linestyle = ':')
        
    return curve
