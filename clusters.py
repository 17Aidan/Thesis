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



# labelClusters
# Input:
# data, and labels associated with data
# 
# as kmeans randomly assigns labels to its clusters, it is the job of labelCLusters to find which real label is associated with the randomly generated ones. It does this by first creating a dictionary (tDict) that observes how many times each real label occurs with a randomly generated one. Another dictionary (sDict) is used to find which label is most associated with which random value. 
# 
# returns: sDict gets returned.

def labelClusters(klabels, labels):
    """"
    labelClusters Input: data, and labels associated with data

as kmeans randomly assigns labels to its clusters, it is the job of labelCLusters to find which real 
label is associated with the randomly generated ones. It does this by first creating a dictionary (tDict) that 
observes how many times each real label occurs with a randomly generated one. Another dictionary (sDict) is used 
to find which label is most associated with which random value.

returns: sDict gets returned.

    """
   
    #print(klabels)
    #print(labels)
   
    tDict = {}
    sDict = {}
    gSList = set(labels)
    glist = []
    
    if len(set(klabels)) != len(gSList):
        return "ERROR: number of clusters does not match with number of labels!"
    
#make dictionary of unique labels associated with clusters to later measure which is more associated with what clusters
    for x in gSList:
        glist.append(x)
        for l in range(len(gSList)):
            tDict[str(x) + str(l)] = 0 #initialize total dictionary
            sDict[x] = 0
            
#put the actual values into dictionary that counts the number of cluster instances for each    
    
    for b in range(len(labels)): 
        for t in tDict:
            if t[:len(t)-1] == str(labels[b]) and t[len(t)-1:] == str(klabels[b]):
                tDict[t] = tDict.get(t) + 1
    print(tDict)

    while len(glist) > 1:
        for x in range(len(gSList)-1):
            a = glist[0]
            for y in range(1, len(glist)):
                if tDict[str(a) + str(x)] < tDict[str(glist[y]) + str(x)]:
                    a = glist[y]
                sDict[a] = x
            glist.remove(a)
    sDict[glist[0]] = len(gSList)-1
    
                
    return sDict
    

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
    
    score = sklearn.metrics.rand_score(klabels, labels)
    
    return score 

