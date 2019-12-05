import numpy as np
import pandas as pd



def x_star_s_omega(x_star_omega_lambd, lo=0, hi=1e2, prec=1e-8, max_iter=10000):
    '''
    Given x_star(Omega, lambd), find constrained RB solution x_star(S, Omega)
    and optimal Lagrange multiplier lambd_star.
    
    input:
    x_star_omega_lambd: optimal constrained solution with some specific Lagrange multiplier;
                        it should be a function only of lambd.
    lo, hi: lower and upper searching bound of lambd_star, both in float
    prec: precision of the normalization condition of x_star(S, Omega), in float
    max_iter: max iteration number, in int
    
    return: {'lambda_star':lambd, 'x_star_s_omega':x} in dictionary
    '''
    
    lambd = (lo+hi)/2
    x = x_star_omega_lambd(lambd)
    iter_num = 0
    
    while np.abs(np.sum(x)-1) > prec:
        # if time out, return np.nan and exit.
        if iter_num >= max_iter:
            return {'lambda_star':np.nan, 'x_star_s_omega':np.nan}
        
        # otherwise follow bisection rule.
        if np.sum(x) < 1: lo = lambd
        else: hi = lambd
        
        # update lambda, x and iteration number.
        lambd = (lo+hi)/2
        x = x_star_omega_lambd(lambd)
        iter_num += 1
    
    return {'lambda_star':lambd, 'x_star_s_omega':x}



def rb_constrained_ccd(lambd, omega, b, c, mu, r, sig, x_init, prec=1e-8, max_iter=10000):
    '''
    Given lambd, find (unnormalized) constrained RB portfolio for this lambd.
    
    input:
    lambd: value of Lagrange multiplier, in float
    omega: constrained formatted by {str(element index): [lower_bound, upper_bound]}, in dict
    b: risk budget (sum up to 1), in np.ndarray with shape=(n, 1)
    c: preference coefficient between return and volatility, in float
    mu: expected annual return vector, in np.ndarray with shape=(n, 1)
    r: risk-free rate, in float
    sig: expected annual covariance matrix, in np.ndarray with shape=(n, n)
    x_init: initial dollar position, GMV=1, in np.ndarray with shape=(n, 1)
    prec: precision for the algo to achieve, in float
    max_iter: max iteration number, in int
    
    return: constrained RB portfolio for this lambd, in np.ndarray with shape=(n, n)
    '''
    
    n = len(x_init)
    pi = mu-r
    iter_num = 0
    
    # first update once in order to make x_new and x_prev different.
    x_prev = x_init.copy()
    x_new = x_prev.copy()
    
    for i in range(n):
        # since vol changes after modifying each coordinate,
        # we need to recompute vol within each loop.
        sig_x = np.sqrt(float(np.dot(np.dot(x_new.T, sig), x_new)))
        
        # coordinate gradient descent
        alpha_i = c*sig[i][i]
        beta_i = c*np.sum(x_new*sig[i, :].T)-c*x_new[i]*sig[i][i] - pi[i]*sig_x
        gamma_i = -lambd*b[i]*sig_x
        
        x_new[i] = (-beta_i+np.sqrt(beta_i**2-4*alpha_i*gamma_i)) / (2*alpha_i)
        
        # apply standard projection operation.
        lo_bound, up_bound = omega[str(i)]
        if x_new[i] <= lo_bound: x_new[i] = lo_bound
        elif x_new[i] >= up_bound: x_new[i] = up_bound
        else: pass
        
    iter_num += 1
    
    # CCD algorithm
    while np.sum(np.abs(x_new-x_prev)) > prec:
        # if max iter is reached, report and return.
        if iter_num > max_iter:
            return 'time out.'
        
        # run CCD for one round.
        x_prev = x_new
        x_new = x_new.copy()
        
        for i in range(n):
            # since vol changes after modifying each coordinate,
            # we need to recompute vol within each loop.
            sig_x = np.sqrt(float(np.dot(np.dot(x_new.T, sig), x_new)))
            
            # coordinate gradient descent
            alpha_i = c*sig[i][i]
            beta_i = c*np.sum(x_new*sig[i, :].T)-c*x_new[i]*sig[i][i] - pi[i]*sig_x
            gamma_i = -lambd*b[i]*sig_x

            x_new[i] = (-beta_i+np.sqrt(beta_i**2-4*alpha_i*gamma_i)) / (2*alpha_i)

            # standard projection operation
            lo_bound, up_bound = omega[str(i)]
            if x_new[i] <= lo_bound: x_new[i] = lo_bound
            elif x_new[i] >= up_bound: x_new[i] = up_bound
            else: pass
        
        iter_num += 1
    
    return x_new



# standard projection operation for box constraints.
def std_proj_box_constraint(v, omega):
    '''
    implement standard projection operation for a vector
    in the case of box constraints.
    
    input:
    v: input vector to be projected, in np.ndarray with shape=(n, 1)
    omega: constrained formatted by {str(element index): [lower_bound, upper_bound]}, in dict
    
    return: projected vector, in np.ndarray with shape=(n, 1)
    '''
    v_proj = v.copy()
    
    for i in range(len(v)):
        # standard projection operation
        lo_bound, up_bound = omega[str(i)]
        if v_proj[i] <= lo_bound: v_proj[i] = lo_bound
        elif v_proj[i] >= up_bound: v_proj[i] = up_bound
        else: pass
    
    return v_proj



def rb_constrained_admm_ccd(lambd, phi, omega, b, c, mu, r, sig, x_init, prec_admm=1e-8, prec_ccd=1e-8,\
                            max_iter_ccd = 1000, k_max=1000):
    '''
    Given lambd, find (unnormalized) constrained RB portfolio for this lambd using ADMM-CCD algorithm.
    
    input:
    lambd: value of Lagrange multiplier, in float
    phi: momentum factor, in float (value typically between (0, 1))
    omega: constrained formatted by {str(element index): [lower_bound, upper_bound]}, in dict
    b: risk budget (sum up to 1), in np.ndarray with shape=(n, 1)
    c: preference coefficient between return and volatility, in float
    mu: expected annual return vector, in np.ndarray with shape=(n, 1)
    r: risk-free rate, in float
    sig: expected annual covariance matrix, in np.ndarray with shape=(n, n)
    x_init: initial dollar position, GMV=1, in np.ndarray with shape=(n, 1)
    prec_admm: precision for the ADMM part to achieve, in float
    prec_ccd: precision for the CCD part to achieve, in float
    max_iter_ccd: max iteration number for CCD part, in int
    k_max: max iteration number for ADMM part, in int
    
    return: constrained RB portfolio for this lambd, in np.ndarray with shape=(n, n)
    '''
    # initialize x, z, and u.
    x_prev = x_init.copy()
    z_prev = x_init.copy()
    u_prev = np.zeros(x_init.shape)
    
    n = len(x_init)
    pi = mu-r
    
    # admm algo frameworks
    for k in range(k_max):
        
        # step 1: x-update
        v_x_new = z_prev - u_prev
        x_tilde = x_prev.copy()
        iter_num_ccd = 0
        
        while True:
            # see if max iter num for ccd is reached.
            if iter_num_ccd >= max_iter_ccd:
                return 'ccd time out'
            
            # use ccd for x-update.
            x_tilde_prev = x_tilde.copy()
            
            for i in range(n):
                sig_x = np.sqrt(float(np.dot(np.dot(x_tilde.T, sig), x_tilde)))
                
                alpha_i = c*sig[i][i] + phi*sig_x
                beta_i = c*np.sum(x_tilde*sig[i, :].T)-c*x_tilde[i]*sig[i][i] - (pi[i]+phi*v_x_new[i])*sig_x
                gamma_i = -lambd*b[i]*sig_x
                
                x_tilde[i] = (-beta_i+np.sqrt(beta_i**2-4*alpha_i*gamma_i)) / (2*alpha_i)
            
            if np.sum(np.square(x_tilde_prev-x_tilde)) <= prec_ccd: break
            iter_num_ccd += 1
        
        x_new = x_tilde.copy()
        
        # step 2: z-update
        # use standard projection for z update.
        v_z_new = x_new + u_prev
        z_new = std_proj_box_constraint(v_z_new, omega)
        
        # step 3: u-update
        u_new = u_prev + x_new - z_new
        
        # step 4: convergence test
        if np.sum(np.square(x_new-z_new)) <= prec_admm:
            return x_new
        
        x_prev, z_prev, u_prev = x_new.copy(), z_new.copy(), u_new.copy()
        
    return 'admm step time out.'



