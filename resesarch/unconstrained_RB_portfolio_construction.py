import numpy as np
import pandas as pd
import datetime
from numpy.random import randint


def target_func(x, r, mu, c, Sig, b, lambd):
    '''
    calculate the value of target function.
    
    input:
    x: dollar position, GMV=1, in np.ndarray with shape=(n, 1)
    r: risk-free rate, in float
    mu: expected annual return vector, in np.ndarray with shape=(n, 1)
    c: preference coefficient between return and volatility, in float
    Sig: expected annual covariance matrix, in np.ndarray with shape=(n, n)
    b: risk budget (sum up to 1), in np.ndarray with shape=(n, 1)
    lambd: Lagrange multiplier, in float
    
    return: target function value, in float
    '''
    # if some element of x is negative, return np.inf.
    if np.sum(x>0) < x.shape[0]:
        return np.inf

    # else, return the corresponding result.
    R = -np.dot(x.T, mu-r) + c*np.sqrt(np.dot(x.T, np.dot(Sig, x)))
    return R - lambd*np.sum(b*np.log(x))



def CD_single_index(i, r, mu, c, Sig, b, lambd, x):
    '''
    helper function: gradient descent for one single index.
    
    input:
    i: index we want to run gradient descent on, in int
    r: risk-free rate, in float
    mu: expected annual return vector, in np.ndarray with shape=(n, 1)
    c: preference coefficient between return and volatility, in float
    b: risk budget (sum up to 1), in np.ndarray with shape=(n, 1)
    lambd: Lagrange multiplier, in float
    x: original dollar position, GMV=1, in np.ndarray with shape=(n, 1)
    
    return: updated dollar position, GMV=1, in np.ndarray with shape=(n, 1)
    ''' 
    # compute the two terms on the numerator separately.
    term1 = 0
    for j in range(mu.shape[0]):
        if j == i:
            continue
        else:
            term1 += x[j]*Sig[i][j]
    term1 *= (-c)

    pi = mu-r
    sig = np.sqrt(float(np.dot(x.T, np.dot(Sig, x))))
    term1 += pi[i]*sig

    term2 = np.sqrt(np.square(term1)+4*lambd*c*b[i]*Sig[i][i]*sig)
    
    # update index i element in x and return.
    res = np.array(x)
    res[i] = (term1+term2)/(2*c*Sig[i][i])
    return res


def RB_unconstrianed(r, mu, c, Sig, b, lambd=1, eta=0.1, opt_algo='CCD', max_iter=10000, prec=1e-6):
    '''
    construct unconstrained risk budgeting portfolio.
    
    input:
    r: risk-free rate, in float
    mu: expected annual return vector, in np.ndarray with shape=(n, 1)
    c: preference coefficient between return and volatility, in float
    Sig: expected annual covariance matrix, in np.ndarray with shape=(n, n)
    b: risk budget (sum up to 1), in np.ndarray with shape=(n, 1)
    lambd: Lagrange multiplier, in float
    eta: learning rate in Newton's algo, in float
    opt_algo: algorithm we want to use, in one of ['Newton', 'CCD', 'RCD']
    max_iter: max iteration number, in int
    prec: precision for the algo to achieve, in float
    
    return: {optimal dollar weight, final value of target function, iteration number it takes, running time in microseconds}
    '''
    # check if optimization algo type is one of these three.
    # if not, raise error.
    if opt_algo not in ['Newton', 'RCD', 'CCD']:
        raise ValueError('Optimization algo type is not correct.')
    
    # get starting time point.
    start_time = datetime.datetime.now()
    
    
    # Newton algorithm
    if opt_algo == 'Newton':
        # starting point of x: all elements are 1.
        x_next = np.ones(shape=(mu.shape[0], 1))
        
        # descent in a quadratic way step by step.
        for iter in range(max_iter):
            x_prev = x_next
            
            # we will compute Sigma*x and store it
            # since we will use it many times.
            Sig_mul_x = np.dot(Sig, x_prev)
            
            # compute sigma=x.T*Sigma*x
            # since we will use it many times.
            sig = np.sqrt(float(np.dot(x_prev.T, Sig_mul_x)))
            
            # compute gradient and Hessian at x_prev.
            grad = -(mu-r) + c*Sig_mul_x/sig - lambd*b/x_prev
            
            hessian = c * (Sig*sig-np.dot(Sig_mul_x, Sig_mul_x.T)) / np.square(sig)\
                      + lambd*np.diag(b/np.square(x_prev))
            
            # descent for one time.
            x_next = x_prev-eta*np.dot(np.linalg.inv(hessian), grad)
            
            # if two adjacent results are close, return.
            target_val_prev = target_func(x_prev, r, mu, c, Sig, b, lambd)
            target_val_next = target_func(x_next, r, mu, c, Sig, b, lambd)
            
            if np.abs((target_val_next-target_val_prev)/target_val_prev) <= prec:
                end_time = datetime.datetime.now()
                return {'x_opt':x_next, 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}
        
        # else stop running and return time-out indicator.
        end_time = datetime.datetime.now()
        return {'x_opt':'time out', 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}
    
    
    # RCD algorithm
    if opt_algo == 'RCD':
        
        asset_num = mu.shape[0]  # number of assets
        # starting point of x: all elements are 1.
        x_next = np.ones(shape=(asset_num, 1))
        
        for iter in range(max_iter):
            # traget function value before this round of descending
            target_val_prev = target_func(x_next, r, mu, c, Sig, b, lambd)
            
            # run coordinate gradient descent for asset_num times.
            for _ in range(asset_num):
                opt_idx = randint(low=0, high=asset_num)
                x_prev = x_next
                x_next = CD_single_index(opt_idx, r, mu, c, Sig, b, lambd, x_prev)
            
            # target function value after this round of descending
            target_val_next = target_func(x_next, r, mu, c, Sig, b, lambd)
            
            # if two adjacent results are close, return.
            if np.abs((target_val_next-target_val_prev)/target_val_prev) <= prec:
                end_time = datetime.datetime.now()
                return {'x_opt':x_next, 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}
        
        # else stop running and return time-out indicator.
        end_time = datetime.datetime.now()
        return {'x_opt':'time out', 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}
    
    
    # CCD algorithm
    if opt_algo == 'CCD':
        
        asset_num = mu.shape[0]  # number of assets
        # starting point of x: all elements are 1.
        x_next = np.ones(shape=(asset_num, 1))
        
        for iter in range(max_iter):
            # target function value before this round of descending
            target_val_prev = target_func(x_next, r, mu, c, Sig, b, lambd)
            
            # run coordinate gradient descent for each index.
            for idx in range(asset_num):
                x_prev = x_next
                x_next = CD_single_index(idx, r, mu, c, Sig, b, lambd, x_prev)
            
            # target function value after this round of descending
            target_val_next = target_func(x_next, r, mu, c, Sig, b, lambd)
            
            # if two adjacent results are close, return.
            if np.abs((target_val_next-target_val_prev)/target_val_prev) <= prec:
                end_time = datetime.datetime.now()
                return {'x_opt':x_next, 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}
        
        # else stop running and return time-out indicator.
        end_time = datetime.datetime.now()
        return {'x_opt':'time out', 'target value':target_val_next, 'iter_num':iter, 'time in mu s':(end_time-start_time).microseconds}