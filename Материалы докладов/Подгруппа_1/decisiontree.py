#!/usr/bin/env python3

from math import log
from collections import defaultdict
import json
import pprint


def calculateEntropy(dataset): #H(X) = - ∑i PX(xi) * log2 (PX(xi))
    counter= defaultdict(int)   
    for record in dataset:      
        label = record[-1]       
        counter[label] += 1
    entropy = 0.0
    for key in counter:
        probability = counter[key]/len(dataset)             
        entropy -= probability * log(probability,2)       
    return entropy

def splitDataset(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset): #featEntrop = - Px0 * log2 (Px0) - Px1 * log2 (Px1)
    baseEntropy = calculateEntropy(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    
    numFeat = len(dataset[0]) - 1            
    for indx in range(numFeat):           
        featValues = {record[indx] for record in dataset}     
        featEntropy = 0.0
        for value in featValues:
            subDataset = splitDataset(dataset, indx, value)      
            probability = len(subDataset)/float(len(dataset))
            featEntropy += probability * calculateEntropy(subDataset) 

        infoGain = baseEntropy - featEntropy   
        if infoGain > bestInfoGain:             
            bestInfoGain = infoGain             
            bestFeature = indx
    return bestFeature                          


def createTree(dataset, features):
    labels = [record[-1] for record in dataset]
    
    
    if labels.count(labels[0]) == len(labels):   
        return labels[0]            
   
    if len(dataset[0]) == 1:                     
        mjcount = max(labels,key=labels.count)  
        return (mjcount) 
    
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = features[bestFeat]
    featValues = {record[bestFeat] for record in dataset}     
    subLabels = features[:]             
    del(subLabels[bestFeat])            
    
    myTree = {bestFeatLabel:{}}         
    for value in featValues:
        subDataset = splitDataset(dataset, bestFeat, value)
        subTree = createTree(subDataset, subLabels)
        myTree[bestFeatLabel].update({value: subTree})  
    return myTree                            

def pprintTree(tree):
    tree_str = json.dumps(tree, indent=4)
    tree_str = tree_str.replace("\n    ", "\n")
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace(',', "")
    tree_str = tree_str.replace("{", "")
    tree_str = tree_str.replace("}", "")
    tree_str = tree_str.replace("    ", " | ")
    tree_str = tree_str.replace("  ", " ")    
    print (tree_str)


def createDataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'],
               [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no'],
               [1, 1, 'maybe'], [0, 0, 'maybe']]
    
    features = ['non-surfacing','flippers']
    label = ['isfish']
    return dataset, features
# non-surfacing     flippers      isfish  
#===============   ==========    ========
#   True(1)         True(1)        yes
#   True(1)         True(1)        yes
#   
#  True(1)         False(0)       no
#   False(0)        True(1)        no
#  False(0)        True(1)        no
#   
#  True(1)         True(1)        maybe
#  False(1)        False(0)       maybe
    
def main():
    dataset, features = createDataset()
    tree = createTree(dataset, features)
    pprintTree (tree) 
    

if __name__ == "__main__":
    main()
