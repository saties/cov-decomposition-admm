# -*- coding: utf-8 -*-
"""
    Created on Sun Jan 28 16:39:43 2018
    
    Covariance Matrix Decomposition
    
    @author:  Satie
"""

import numpy as np
from numpy.linalg import matrix_rank

from multiprocessing import Pool

class Decomposer(object):
    
    def __init__(self, data, preavg, delta):
        
        self.__data  = data
        self.__sigma = preavg
        self.__N, self.__p = data.shape   # N by p
        self.delta = delta                # observation frequency
    
    
    def __l1_norm_off_diag(self, matrix):
        
        matrix_off = np.abs(matrix - np.diag(np.diag(matrix)))
        
        return matrix_off.sum()
    
    
    def __Obj(self, S, F, T, lam):
        
        err = 0.5 * (np.linalg.norm(S - F - T, 'fro') ** 2)
        
        reg_F = lam[0] * np.linalg.norm(F, 'nuc')
        reg_T  = lam[1] * self.__l1_norm_off_diag(T)
        
        return err + reg_F + reg_T
    
    
    def __Lag(self, S, F, T, F_cp, T_cp, LAM1, LAM2, mu, lam):
        
        penF = ((2 * mu[0]) ** (-1)) * (np.linalg.norm(F - F_cp, 'fro') ** 2)
        penT = ((2 * mu[1]) ** (-1)) * (np.linalg.norm(T - T_cp, 'fro') ** 2)
        
        duaF = (LAM1 * (F_cp - F)).sum()
        duaT = (LAM2 * (T_cp - T)).sum()
        
        return self.__Obj(S, F, T, lam) + penF + penT + duaF + duaT
    
    
    def Proj_SoftThres(self, matrix, eps):
        
        _p, _q = matrix.shape
        
        assert _p == _q
        
        diag = matrix * np.eye(_p)
        off  = matrix - diag
        truc = np.abs(off) - eps
        sign = np.sign(off)
        truc[truc < 0] = 0
        sign[truc < 0] = 0
        
        return sign * truc + diag
    
    
    def Proj_SVT(self, matrix, eps):
        
        _p, _q = matrix.shape
        
        assert _p == _q
            
        s, V = np.linalg.eig(matrix)
        s = s - eps
        s[s <= 0] = 0
        
        return np.dot(V, np.dot(np.diag(s), V.T))
    
    
    def Proj_PD(self, matrix, eps):
        
        _p, _q = matrix.shape
        
        assert _p == _q
            
        e, V = np.linalg.eig(matrix)
        
        # Handle complex eigenvalues due to numerical rounding.
        
        if isinstance(e[0], complex):
            
            if np.allclose(matrix, matrix.T):
                
                e = np.real(e)
                V = np.real(V)
                
            else:
                
                raise ValueError('Proj_PD: Complex eigens encountered.')
            
        e[e < eps] = eps
        
        return np.dot(V, np.dot(np.diag(e), V.T))
    

    def GCV_Alt(self, L, S, lam1, eps, dof = 1):
        
        # Temp version

        def DoF1(eigens, lam):

            b, V = np.linalg.eig(L)
            b = np.real(b)

            sig2 = np.outer(eigens, np.ones(len(eigens)))
            deno = (sig2 - sig2.T)
            np.fill_diagonal(deno, np.inf)
            deno[deno == 0] = np.inf
            assert np.all(deno != 0)
            deno = deno ** -1
            
            cons = np.sqrt(eigens) * (np.sqrt(eigens) - lam)
            
            dof  = 1 + 2 * cons * deno.sum(0)
            ind  = (b > 0).astype('int')

            return np.dot(1 + 2 * dof, ind)

        def DoF2(eigens, lam):

            ind = (eigens >= lam).astype('int'); res = 0

            for i in range(ind.sum()):

                res1 = 0; res2 = 0

                for jj in range(len(eigens)):

                    if jj != i:

                        res2 += eigens[i] / eigens[i] - eigens[jj]

                    if jj > ind.sum():

                        res1 += eigens[jj] / eigens[i] - eigens[jj]

                res += res1 - 2 * lam * res2

            return (2 * len(eigens) - ind.sum()) * ind.sum() + res
        
        s, V = np.linalg.eig(self.__sigma - S)
        s[s < eps] = eps; s = np.real(s) # Note s is already sorted.

        if dof == 1:

            df1 = DoF1(s, lam1)

        else:

            df1 = DoF2(s, lam1)

        tot_df = df1 + np.count_nonzero(S)
        err    = np.linalg.norm(self.__sigma - L - S, 'fro')
        aic    = np.log(err) + (2 * tot_df) / (self.__p ** 2)

        if self.__p ** 2 <= tot_df:

            gcv = 999999

        else:

            gcv = err / (self.__p ** 2 - tot_df)

        return gcv, aic


    def __Initializer(self, S, mode, lam, verbose = False):
        
        _p, _q = S.shape
        
        if mode == 'SP':
            
            res = self.Proj_SoftThres(S, lam)
            
        elif mode == 'LR':
            
            res = self.Proj_SVT(S, lam)
        
        elif mode == 'random':
            
            res = np.random.uniform(S.min(), S.max(), size = _p * _p)
            res = res.reshape((_p, _p))
            res = 0.5 * (res + res.T)
            
        else:
            
            res = np.zeros_like(S)
            
        return res

    
    def Solver_ADMM(self, lam, verbose = 2, args_dict = {}):
    
    # def Solver_ADMM(self, params):

    #     lam, verbose, args_dict = map(params.get, ['lam', 'verbose', 'args_dict'])
        
        params = {'tol': 1e-4, 'max_iter': 200,
                  'eps': 1e-4, 'mu': (2, 2), 'monitor': 1}
        
        params.update(args_dict)
        
        _S = self.__sigma
        _p, _q = _S.shape
        
        assert _p == _q
            
        # Initialize.
        
        if verbose >= 2:
            
            print('------------------------------------')
            print('Solver_ADMM: Initializing.')
        
        lam1, lam2 = lam
        mu1,  mu2  = params['mu']
        
        LAM1 = np.zeros((_p, _p))
        LAM2 = np.zeros((_p, _p))
        
        F = self.__Initializer(0.5 * _S, 'LR', lam1)
        T = self.__Initializer(0.5 * _S, 'SP', lam2)
        
        epoch = 1; converge = False; err = np.linalg.norm(_S - F - T, 'fro')
        
        while (epoch <= params['max_iter']) and (not converge):
            
            if verbose == 2:
               
                print('Epoch: {}'.format(epoch))
            
            last_F = F; last_T = T

            if params['monitor'] >= 2:

                last_e, last_V = np.linalg.eig(last_F)
            
            # Low-rank: Projection.
            F_cp = self.Proj_PD(F + mu1 * LAM1, 0.)
            
            # Low-rank: Main update.
            F = (1 + mu1) ** (-1) * self.Proj_SVT(mu1 * (_S - T - LAM1) + F_cp, lam1 * mu1)
            
            # Low-rank: Dual update.
            LAM1 = LAM1 + mu1 ** (-1) * (F - F_cp)
            
            # Sparse: Projection.
            T_cp = self.Proj_PD(T + mu2 * LAM2, params['eps'])
            
            # Sparse: Main update.
            T = (1 + mu2) ** (-1) * self.Proj_SoftThres(mu2 * (_S - F - LAM2) + T_cp, lam2 * mu2)
            
            # Sparse: Dual update.
            LAM2 = LAM2 - mu2 ** (-1) * (T_cp - T)
            
            # Post processing.
            
            epoch += 1

            if params['monitor'] >= 2:

                cur_e, cur_V = np.linalg.eig(F)

                err  = np.linalg.norm(cur_e - last_e, 2)
                err += np.linalg.norm(cur_V - last_V, 'fro')
                err += np.linalg.norm(T - last_T, 'fro')

            else:

                err = np.linalg.norm(last_F + last_T - F - T, 'fro')

            if verbose >= 2:
                
                print('Solver_ADMM: Frobenius error: {}.'.format(
                        np.linalg.norm(_S - F - T, 'fro')))

                print('Solver_ADMM: Objective value: {}.'.format(
                        self.__Obj(_S, F, T, lam)))

                print('Solver_ADMM: Lag value: {}.'.format(
                        self.__Lag(_S, F, T, F_cp, T_cp, LAM1, LAM2, params['mu'], lam)))
            
            if np.abs(err) < params['tol']:
                
                converge = True

                if verbose:

                    print('Solver_ADMM: Converged with achieved tol {},'.format(err))
                
        if epoch > params['max_iter'] and verbose:
            
            print('Solver_ADMM: Maximum iteration {} reached.'.format(params['max_iter']))
        
        return F, T 
    

    def __D(self, F, T, F_next, T_next):
        
        return np.linalg.norm(F - F_next, 'fro') + np.linalg.norm(T - T_next, 'fro')
    

    def Estimator(self, params_lam, params_gam, solver_args_dict = {},
                  verbose = 3, grid = False, use = 'GCV'):

        # Fixme: Reduce args

        solver_args = {'tol': 1e-3, 'max_iter': 100, 'eps': 1e-4, 'monitor': 1}

        solver_args.update(solver_args_dict); solver_args['eps']
    
        # Low rank penalty

        if params_lam is None:
            
            params_lam = (-2, 2, 20)

        # Sparse penalty    
        
        if params_gam is None:
            
            params_gam = (-2, 2, 20)
            
        lam_try = 10 ** (np.linspace(*params_lam))
        gam_try = 10 ** (np.linspace(*params_gam))
        nl, ng  = (params_lam[2], params_gam[2])
                
        D = {'GCV': {'lam1': np.zeros(nl), 'lam2': np.zeros(ng)},
             'AIC': {'lam1': np.zeros(nl), 'lam2': np.zeros(ng)}}

        # Deal with use

        if len(use) == 4:

            dof = int(use[-1])
            use = use[:3]

            assert dof <= 2 and dof > 0
            assert use in ['GCV', 'AIC']

        elif use in ['GCV', 'AIC']:

            dof = 1

        else:

            raise ValueError()
        
        # Select lambda

        for l in range(nl):
            
            if verbose:

                print("Estimator: Tuning lambda {} / {}".format(l + 1, nl))
            
            lam_cur = (lam_try[l], gam_try[0])

            f, t = self.Solver_ADMM(lam_cur, verbose - 1, solver_args)

            D['GCV']['lam1'][l], D['AIC']['lam1'][l] = self.GCV_Alt(f, t, lam_cur[0], eps, dof)

            if verbose:

                print("Estimator: Current GCV {}".format(D['GCV']['lam1'][l]))
                print("Estimator: Current AIC {}".format(D['AIC']['lam1'][l]))
            
        lam_final = lam_try[D[use]['lam1'].argmin()] # Fixme: possible duplication

        # Select gamma for lam_final

        for g in range(ng):

            if verbose:

                print("Estimator: Tuning gamma {} / {}".format(g + 1, nl))
            
            lam_cur = (lam_final, gam_try[g])
                
            f, t = self.Solver_ADMM(lam_cur, verbose - 1, solver_args)
                
            D['GCV']['lam2'][g], D['AIC']['lam2'][g] = self.GCV_Alt(f, t, lam_cur[0], eps)

            if verbose:
                   
                print("Estimator: Current GCV {}".format(D['GCV']['lam2'][g]))
                print("Estimator: Current AIC {}".format(D['AIC']['lam2'][g]))

        gam_final = gam_try[D[use]['lam2'].argmin()]
 
        lam_final_pair = (lam_final, gam_final)
            
        if verbose:

            print("Finalizing Results.")
            print("Selected lam: {}".format(lam_final_pair))
            print("Best {}: {}".format(use, D[use]['lam2'].min()))
        
        # Finalize
        
        f, t = self.Solver_ADMM(lam_final_pair, verbose - 1, solver_args)
        
        if grid:
            
            return f, t, D, lam_final_pair
        
        else:
            
            return f, t


    def Estimator_Parallel_FullGrid(self, params_lam, params_gam, npool,
                                    solver_args_dict = {}, grid = False,
                                    use = 'GCV'):

        # M by N grid search. Parallel computation over N for each m in [M].

        solver_args = {'tol': 1e-3, 'max_iter': 100, 'eps': 1e-4, 'monitor': 1}

        solver_args.update(solver_args_dict); eps = solver_args['eps']
    
        # Low rank penalty

        if params_lam is None:
            
            params_lam = (-2, 2, 20)

        # Sparse penalty    
        
        if params_gam is None:
            
            params_gam = (-2, 2, 20)
            
        lam_try = 10 ** (np.linspace(*params_lam))
        gam_try = 10 ** (np.linspace(*params_gam))

        D = {'GCV': np.zeros((len(lam_try), len(gam_try))),
             'AIC': np.zeros((len(lam_try), len(gam_try)))}

        # Deal with use

        if len(use) == 4:

            dof = int(use[-1])
            use = use[:3]

            assert dof <= 2 and dof > 0
            assert use in ['GCV', 'AIC']

        elif use in ['GCV', 'AIC']:

            dof = 1

        else:

            raise ValueError()

        for l in range(len(lam_try)):

            pool = Pool(npool)
            iteg = [((lam_try[l], g), 0, solver_args) for g in gam_try]
            ft = pool.starmap(self.Solver_ADMM, iteg)

            pool.terminate()

            for g in range(len(gam_try)):

                D['GCV'][l, g], D['AIC'][l, g] = self.GCV_Alt(ft[g][0], ft[g][1],
                                                              lam_try[l], eps, dof)

        best_pos = np.unravel_index(D[use].argmin(), D[use].shape)
        lam_final_pair = (lam_try[best_pos[0]], gam_try[best_pos[1]])

        print("Finalizing Results.")
        print("Selected lam: {}".format(lam_final_pair))
        print("Best {}: {}".format(use, D[use][best_pos]))

        print('-' * 30)        
        
        f, t = self.Solver_ADMM(lam_final_pair, 2, solver_args)

        if grid:
            
            return f, t, D, lam_final_pair
        
        else:
            
            return f, t


    def Estimator_Parallel_Simplified(self, params_lam, params_gam, npool,
                                      solver_args_dict = {}, grid = False,
                                      use = 'GCV'):

        # Offer M + N grid search.

        solver_args = {'tol': 1e-3, 'max_iter': 100, 'eps': 1e-4, 'monitor': 1}

        solver_args.update(solver_args_dict); eps = solver_args['eps']
    
        # Low rank penalty

        if params_lam is None:
            
            params_lam = (-2, 2, 20)

        # Sparse penalty    
        
        if params_gam is None:
            
            params_gam = (-2, 2, 20)
            
        lam_try = 10 ** (np.linspace(*params_lam))
        gam_try = 10 ** (np.linspace(*params_gam))
        nl, ng  = (params_lam[2], params_gam[2])
                
        D = {'GCV': {'lam1': np.zeros(nl), 'lam2': np.zeros(ng)},
             'AIC': {'lam1': np.zeros(nl), 'lam2': np.zeros(ng)}}

        # Deal with use

        if len(use) == 4:

            dof = int(use[-1])
            use = use[:3]

            assert dof <= 2 and dof > 0
            assert use in ['GCV', 'AIC']

        elif use in ['GCV', 'AIC']:

            dof = 1

        else:

            raise ValueError()
        
        # Select lambda

        pool = Pool(npool)
        ite1 = [((l, gam_try[0]), 0, solver_args) for l in lam_try]
        ft   = pool.starmap(self.Solver_ADMM, ite1)
        # ite1 = [{'lam': (l, gam_try[0]), 'verbose': 0, 'args_dict': solver_args} for l in lam_try]
        # ft   = pool.map(self.Solver_ADMM, ite1)

        pool.terminate()

        for l in range(nl):
            
            D['GCV']['lam1'][l], D['AIC']['lam1'][l] = self.GCV_Alt(ft[l][0], ft[l][1], lam_try[l], eps, dof)

        lam_final = lam_try[D[use]['lam1'].argmin()] # Fixme: possible duplication

        # Select gamma for lam_final

        pool = Pool(npool)
        ite2 = [((lam_final, g), 0, solver_args) for g in gam_try]
        ft   = pool.starmap(self.Solver_ADMM, ite2)
        # ite2 = [{'lam': (lam_final, g), 'verbose': 0, 'args_dict': solver_args} for g in gam_try]
        # ft   = pool.map(self.Solver_ADMM, ite2)

        pool.terminate()

        for g in range(ng):

            D['GCV']['lam2'][g], D['AIC']['lam2'][g] = self.GCV_Alt(ft[g][0], ft[g][1], lam_final, eps, dof)

        gam_final = gam_try[D[use]['lam2'].argmin()]
 
        lam_final_pair = (lam_final, gam_final)
        
        # Finalize
        
        print("Finalizing Results.")
        print("Selected lam: {}".format(lam_final_pair))
        print("Best {}: {}".format(use, D[use]['lam2'].min()))

        print('-' * 30)        
        
        f, t = self.Solver_ADMM(lam_final_pair, 2, solver_args)

        if grid:
            
            return f, t, D, lam_final_pair
        
        else:
            
            return f, t
