# -*- coding: utf-8 -*-
# @Date    : 2018-07-23 19:26:33
# @Author  : Satie (satie.yao@my.cityu.edu.hk)
# @Link    :  
# @Version : 2.0.0
# 
# Main Utilities

__all__ = ['SwitchCase', 'Partition2dGrids', 'Summarize']

import numpy as np

def SwitchCase(data_dict, case):

    # Note: Add theta to return before passing into PreAverage().

    if case == 4:

        # Async, Noisy

        preavg_params = {'Y':        data_dict['contamY'],
                         'subtract': True,
                         'addpower':  0.}

    elif case == 3:

        # Sync, Noisy

        preavg_params = {'Y':        data_dict['NoisyX'],
                         'subtract': True,
                         'addpower':  0.}

    elif case == 2:

        # Async, Clean

        preavg_params = {'Y':        data_dict['contamX'],
                         'subtract': False,
                         'addpower':  0.}

    elif case == 1:

        # Sync, Clean

        preavg_params = {'Y':        data_dict['TrueX'],
                         'subtract': False,
                         'addpower':  0.}

    else:

        raise ValueError()

    preavg_params.update({'data_dict': data_dict}) # To extract other values.

    return  preavg_params

def Partition1dGrids(linspace_params, nblocks):

    """
    Partition grids into N unordered blocks.
    Return list of nested tuples (params_lam_blocki, params_gam_blocki).
    """

    assert isinstance(linspace_params, tuple) and len(linspace_params) == 3
    assert linspace_params[2] % nblocks == 0
        
    ran = linspace_params[:2]; num = linspace_params[2]; each = num // nblocks
    pnt = np.linspace(ran[0], ran[1], nblocks)
    res = [(pnt[i], pnt[i + 1], each) for i in range(nblocks - 1)]

    return res

def Partition2dGrids(linspace_params1, linspace_params2, shape):

    list1 = Partition1dGrids(linspace_params1, shape[0])
    list2 = Partition1dGrids(linspace_params2, shape[1])

    res = [(a, b) for a in list1 for b in list2]

    return res

def Summarize(sigma, sigma_hat, eps = 1e-4):

    d, _d = sigma.shape
    assert sigma.shape == sigma_hat.shape
    assert d == _d

    err   = sigma_hat - sigma
    err_s = err.T * err
    err_i = np.linalg.inv(sigma_hat) - np.linalg.inv(sigma)

    e, V = np.linalg.eig(sigma)
    e = np.real(e); V = np.real(V)
    e[e < eps] = eps
    sphe = V.dot(np.diag(np.sqrt(e ** -1))).dot(V.T)
    sphi = V.dot(np.diag(np.sqrt(e))).dot(V.T)

    q_norm_err = (d ** -1) * np.linalg.norm(sphe.dot(err).dot(sphe), 'fro')
    # q_norm_inv = np.sqrt(d) ** -1 * np.linalg.norm(sphi.dot(err_i).dot(sphi), 'fro')

    res = {'Fro_norm_conv': np.linalg.norm(err_s, 'fro'),
           'Spe_norm_conv': np.linalg.norm(err_s, 2),
           'Sup_norm_conv': np.abs(err_s).max(),
           'Fro_norm_err':  np.linalg.norm(err, 'fro'),
           'Spe_norm_err':  np.linalg.norm(err, 2),
           'Sup_norm_err':  np.abs(err).max(),
           'Fro_norm_inv':  np.linalg.norm(err_i, 'fro'),
           'Spe_norm_inv':  np.linalg.norm(err_i, 2),
           'Sup_norm_inv':  np.abs(err_i).max(),
           'Sig_norm_err':  q_norm_err}

    return res