import numpy as np
import pandas as pd

def normalize(old_weight):
    '''
    normalize the old weight matrix to make it dollar neutral.
    note that GMV = 1, which means that all positive (negative) weights
    on each day sum up to positive (negative) 0.5.
    
    user case: new_weight = normalize(old_weight)
    '''
    new_weight = old_weight.copy()
    new_weight[new_weight>0] = new_weight.divide(np.sum(np.abs(new_weight[new_weight>0]), axis=1), axis=0)*0.5
    new_weight[new_weight<0] = new_weight.divide(np.sum(np.abs(new_weight[new_weight<0]), axis=1), axis=0)*0.5
    
    return new_weight
    