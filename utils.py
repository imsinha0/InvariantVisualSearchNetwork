
import torch
import torch.nn as nn
import numpy as np


image_positions = {1:(56, 539), 2:(54, 234), 3:(330,85), 4:(592, 230), 5:(584, 530), 6:(337, 675)} #centers

def distance(p1, p2):
    #p1 and p2 are tuples
    return np.max(np.abs(np.array(p1) - np.array(p2)))

def arraySearchProcesswithPath(attentionMap, gtpos):
    numSearches = 0
    searchPath = []

    #convert attentionMap to numpy array
    attentionMap = attentionMap.numpy()

    found = False
    while(not found):
        numSearches += 1

        #print max value in attentionMap, which is a tensor
        print("max value in attentionMap", np.max(attentionMap))

        #get maxpoint as tuple
        maxPoint = np.unravel_index(np.argmax(attentionMap), attentionMap.shape) #y then x
        maxPoint = (maxPoint[1], maxPoint[0]) #x then y

        searchPath.append(maxPoint)

        print("searching at", maxPoint)


        if distance(maxPoint, image_positions[gtpos]) <= 45:
            found = True
        else:
            #set attentionMap to 0 in 45x45 region around maxPoint
            x, y = maxPoint
            x_start = max(0, x - 22)
            x_end = min(attentionMap.shape[1], x + 22)
            y_start = max(0, y - 22)
            y_end = min(attentionMap.shape[0], y + 22)
            attentionMap[y_start:y_end, x_start:x_end] = 0

    return numSearches, searchPath


def naturalSearchProcesswithPath(attentionMap, bbox):
    numSearches = 0
    searchPath = []

    #bbox is tuple of (x_min, y_min, x_max, y_max)

    #convert attentionMap to numpy array
    attentionMap = attentionMap.numpy()

    found = False
    while(not found):
        numSearches += 1

        #print max value in attentionMap, which is a tensor
        print("max value in attentionMap", np.max(attentionMap))

        #get maxpoint as tuple
        maxPoint = np.unravel_index(np.argmax(attentionMap), attentionMap.shape) #y then x
        maxPoint = (maxPoint[1], maxPoint[0]) #x then y

        searchPath.append(maxPoint)

        print("searching at", maxPoint)

        #want to check if maxPoint is within bbox
        if maxPoint[0] >= bbox[0] and maxPoint[0] <= bbox[2] and maxPoint[1] >= bbox[1] and maxPoint[1] <= bbox[3]:
            found = True
        else:
            #set attentionMap to 0 in 200x200 region around maxPoint
            x, y = maxPoint
            x_start = max(0, x - 100)
            x_end = min(attentionMap.shape[1], x + 100)
            y_start = max(0, y - 100)
            y_end = min(attentionMap.shape[0], y + 100)
            attentionMap[y_start:y_end, x_start:x_end] = 0

    return numSearches, searchPath


def waldoSearchProcesswithPath(attentionMap, bbox):
    numSearches = 0
    searchPath = []

    #bbox is tuple of (x_min, y_min, x_max, y_max)

    #convert attentionMap to numpy array
    attentionMap = attentionMap.numpy()

    found = False
    while(not found):
        numSearches += 1

        #print max value in attentionMap, which is a tensor
        print("max value in attentionMap", np.max(attentionMap))

        #get maxpoint as tuple
        maxPoint = np.unravel_index(np.argmax(attentionMap), attentionMap.shape) #y then x
        maxPoint = (maxPoint[1], maxPoint[0]) #x then y

        searchPath.append(maxPoint)

        print("searching at", maxPoint)

        #want to check if maxPoint is within bbox
        if maxPoint[0] >= bbox[0] and maxPoint[0] <= bbox[2] and maxPoint[1] >= bbox[1] and maxPoint[1] <= bbox[3]:
            found = True
        else:
            #set attentionMap to 0 in 100x100 region around maxPoint
            x, y = maxPoint
            x_start = max(0, x - 50)
            x_end = min(attentionMap.shape[1], x + 50)
            y_start = max(0, y - 50)
            y_end = min(attentionMap.shape[0], y + 50)
            attentionMap[y_start:y_end, x_start:x_end] = 0

    return numSearches, searchPath


