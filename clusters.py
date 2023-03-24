import kneed
import sklearn
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from collections import Counter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import matplotlib.pyplot as plt
from collections import defaultdict


# Create function that runs kmeans with a range of clusters with the data, test the accuracy (inertia) of different cluster numbers representing the raw data using the elbow method, return the optimal number of clusters

def K(data):

    inertias = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(range(1, 11), inertias, curve="convex", direction="decreasing")
    return kl.elbow


# Create function that plots kmeans data based on the data and elbow method
def plotKMeans(data, klabels): 
    """"
    input data,
    returns plot of data with kmeans applied to it, differnt colors represent different labels
    
    """
    date = np.array(data)
    x, y = date.T
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.scatter(x, y, c=klabels)
    plt.show()

def labelClusters(klabels, labels, print_df = False):
    
    # create default dictionary with default an empty list
    # the key is the klabel and the value is the list of values for that label
    counters = defaultdict(list)

    # for each klabel and label
    for k, l in zip(klabels, labels) :    
        # add the label to the dictionary
        counters[k] += [l]
    # count the number of values for each klabel
    for k in counters :
        counters[k] = Counter(counters[k])
    # create a dictionary of clusterLabels based on the most common value for each klabel
    clusterLabels = {}
    for k in counters:
        c = counters[k]
        clusterLabels[k] = c.most_common(1)[0][0]
    print(clusterLabels)
    return clusterLabels
    """
     # create a dictionary of counters, one for each klabel
    counters = {k: [] for k in set(klabels)}

    # loop through each label and add it to the corresponding counter
    for k, label in zip(klabels, labels):
        counters[k].append(label)

    # count the number of values for each klabel
    for k in counters:
        counters[k] = Counter(map(tuple, counters[k]))

    # create a dictionary of clusterLabels based on the most common value for each klabel
    clusterLabels = {}
    for k in counters:
        clusterLabels[k] = counters[k].most_common(1)[0][0]

    if print_df:
        df = pd.DataFrame({'label': labels, 'cluster': klabels})
        print(df)

    return clusterLabels
    """


def getClusters(data, labels):
    """"
    input: data, labels
    returns tuple: cluster labels, number of clusters
    
    """
    
    y = K(data)
    gSList = set(labels)
    glist = []
    for x in gSList:
        glist.append(x)
    glist.append(y)
    v = tuple(glist)
    return v


def assignClusters(klabels, labels):
    """"
    input: arbitrary k means labels, true labels
    
    converts arbitrary k means labels list into a list based on which arbitrary cluster is associated with which 
    real cluster (using labelClusters)
    
    returns: list of predicted clusters w/ true labels
    """
    
    tlist = []
    dictt = labelClusters(klabels, labels)
    
    
    for y in range(len(klabels)):
        for x in dictt:
            if dictt[x] == klabels[y]:
                tlist.append(x)
    return tlist


def randIndex(klabels, labels):
    """"
    input: klabels, labels
    
    uses assign clusters and true cluster labels to measure accuracy of kmeans on data. rand index takes compares 
    how well the predicted clusters and the true clusters line up. getClusters is used to find cluster names and 
    predicted number of clusters (k) in final tuple
    
    return rand index
    """
    #listl = assignClusters(klabels, labels)
    print(labels)
    #print(listl)
    print(klabels)
    
    score = sklearn.metrics.rand_score(klabels, labels)
    
    return score 

