
import torch
import torch.nn as nn
import numpy as np


image_positions = {1:(56, 539), 2:(54, 234), 3:(330,85), 4:(592, 230), 5:(584, 530), 6:(337, 675)} #centers

def distance(p1, p2):
    return np.max(np.abs(p1-p2))

def arraySearchProcesswithPath(attentionMap, gtpos):
    numSearches = 0
    searchPath = []

    found = False
    while(not found):
        numSearches += 1

        #get maxpoint as tuple
        maxPoint = np.unravel_index(attentionMap.argmax(), attentionMap.shape)
        searchPath.append(maxPoint)

        if distance(maxPoint, image_positions[gtpos]) <= 45:
            found = True
        else:
            #set attentionMap to 0 in 45x45 region around maxPoint
            x, y = maxPoint
            attentionMap[x-22:x+22, y-22:y+22] = 0

    return numSearches, searchPath



