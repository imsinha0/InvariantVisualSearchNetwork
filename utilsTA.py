
import torch
import torch.nn as nn
import numpy as np
import cv2


def compute_edge_density(img):
    img = np.array(img)  # Ensure it's a NumPy array

    print("Original img shape:", img.shape)
    print("Original img dtype:", img.dtype)

    # Ensure the image is uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Normalize if it's float (e.g., Torch tensor)

    # Convert to grayscale if it's an RGB image (3 channels)
    if len(img.shape) == 3 and img.shape[0] in [3, 4]:  # Channels-first format
        img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) → (H, W, C)
    
    if len(img.shape) == 3 and img.shape[2] in [3, 4]:  # Standard RGB/BGR image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print("Processed img shape:", img.shape)
    print("Processed img dtype:", img.dtype)

    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)

    # Compute edge density
    edge_density = np.sum(edges > 0) / img.size

    return edge_density


def distance(p1, p2):
    #p1 and p2 are tuples
    return np.max(np.abs(np.array(p1) - np.array(p2)))



def naturalSearchProcesswithPathFixedN(attentionMap, maxLooks):
    numSearches = 0
    searchPath = []

    attentionMap = attentionMap.numpy()

    found = False
    while(not found):
        numSearches += 1

        #print max value in attentionMap, which is a tensor
        #print("max value in attentionMap", np.max(attentionMap))

        #get maxpoint as tuple
        maxPoint = np.unravel_index(np.argmax(attentionMap), attentionMap.shape) #y then x
        maxPoint = (maxPoint[1], maxPoint[0]) #x then y

        searchPath.append(maxPoint)

        x, y = maxPoint
        x_start = max(0, x - 100)
        x_end = min(attentionMap.shape[1], x + 100)
        y_start = max(0, y - 100)
        y_end = min(attentionMap.shape[0], y + 100)
        attentionMap[y_start:y_end, x_start:x_end] = 0

        if numSearches >= maxLooks:
            break

    return numSearches, searchPath, found


def naturalSearchProcesswithPathNaiveGaussian(attentionMap, mean, std):

    #bbox is tuple of (x_min, y_min, x_max, y_max)
    #draw from gaussian distribution to determine number of looks
    maxLooks = int(np.random.normal(mean, std))

    return naturalSearchProcesswithPathFixedN(attentionMap, maxLooks)

def naturalSearchProcesswithEdgeDensity(attentionMap, stimuli_img):
    edge_density = compute_edge_density(stimuli_img)
    return naturalSearchProcesswithPathNaiveEvidenceAccumulation(attentionMap, np.exp(-edge_density))


def naturalSearchProcesswithPathNaiveEvidenceAccumulation(attentionMap, threshold):
    #we use e^(-x) as the probability of finding the target at a given location, where x is the attention value
    numSearches = 0
    searchPath = []
    attentionMap = attentionMap.numpy()

    found = False

    currentProb = 0
    while(currentProb < threshold and numSearches < 200 and found == False):
        numSearches += 1

        currentProb += np.exp(-np.max(attentionMap))

        #get maxpoint as tuple
        maxPoint = np.unravel_index(np.argmax(attentionMap), attentionMap.shape) #y then x
        maxPoint = (maxPoint[1], maxPoint[0]) #x then y

        searchPath.append(maxPoint)

        x, y = maxPoint
        x_start = max(0, x - 100)
        x_end = min(attentionMap.shape[1], x + 100)
        y_start = max(0, y - 100)
        y_end = min(attentionMap.shape[0], y + 100)
        attentionMap[y_start:y_end, x_start:x_end] = 0

    return numSearches, searchPath, found


