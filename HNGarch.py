from numpy.random import normal
from math import log, pi, sqrt, exp
from statistics import variance
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

#############
#  GENERAL  #
#############

def llhngarch(par, ts=None, r_f=0., symmetric=False):
    '''

    Parameters
    ----------
    par : list
        List of parameters of the function.
    ts : list, optional
        timeseries of log-returns. The default is None.
    r_f : float, optional
        risk-free interest rate. The default is 0..
    symmetric : bool, optional
        boolean, if True the gamma parameter is set to 0. The default is False.

    Returns
    -------
    float
        returns the negative log-likelihood of the Heston-Nandi GARCH model with the given parameters.

    '''
    
    # init
    h_vec = []
    z_vec = []
    ll = []
    r = r_f
    
    omega = 1/(1+exp(-par[0]))
    alpha = 1/(1+exp(-par[1]))
    beta = 1/(1+exp(-par[2]))
    if not symmetric : gam = par[3] 
    else: gam = 0 
    lam = par[4]
    
    # barrier function to ensure stationarity
    if beta + (alpha * gam * gam) > 1: return 1e50
    
    h = variance(ts)
    h_vec.append(h)
    z = (ts[0] - r - lam * h)/sqrt(h)
    z_vec.append(z)
    
    try:
        l = log(norm.pdf(z)/sqrt(h))
    except:
        l = 1e50
    ll.append(l)
    for i in range(1, len(ts)):
        h = omega + alpha * pow(z - gam * sqrt(h),2) + beta * h
        h_vec.append(h)
        z = (ts[i] - r - lam * h)/sqrt(h)
        z_vec.append(z)
        try:
            l = log(norm.pdf(z)/sqrt(h))
        except:
            l = 1e50
        ll.append(l)
        
    return -1 * sum(ll)

def A_loop(a, b, r_f, phi, omega, alpha):
        a = a + phi * r_f + b *omega - 0.5*np.log(1 - 2 * alpha * b)
        return a
    
def B_loop(b, phi, alpha, beta, gam, lam):
    b = phi * (lam + gam) - 0.5 * gam**2 + beta * b + (0.5*(phi - gam)**2)/(1 - 2 * alpha * b)
    return b       
        
########################
#  Heston-Nandi GARCH  #
########################

class HNGarch(object):
    '''

    Attributes
    ----------
    timeseries : list
        timeseries over which the parameters will be estimated. The default is None.
    r_f : float, optional
        risk-free rate. The default is 0..
    omega : float, optional
        parameter omega of the Heston-Nandi GARCH model, represents the constant. The default is None.
    alpha : float, optional
        parameter alpha of the Heston-Nandi GARCH model, determines the kurtosis of the distribution. The default is None.
    beta : float, optional
        parameter beta of the Heston-Nandi GARCH model. The default is None.
    gamma : TYPE, optional
       parameter gamma of the Heston-Nandi GARCH model, represents the skewness of the distribution. The default is None.
    p_lambda : TYPE, optional
        parameter lambda of the Heston-Nandi GARCH model. The default is None.
    lr_var : TYPE, optional
        long-run variance of the estimated GARCH model. The default is None.
    gamma_star : TYPE, optional
        risk-neutral gamma parameter for Heston-Nandi GARCH forecasting. The default is None.
    h_t0 : TYPE, optional
        terminal value of the variance process of the estimated Heston-Nandi GARCH model. The default is None.
        
    Methods
    -------
    GARCH_fit(self)
        estimate model parameters through maximum-likelihood estimation.
    lr_variance(self)
        computes long-run variance of the process, returns long-run variance.
    lr_vol(self)
        returns long-run volatility of the process.
    get_std_errors(self)
        returns the standard errors of the estimated parameters omega, alpha, beta, gamma, lambda.
    ts_var(self, vec)
        computes estimated GARCH variance process of the time-series, returns last value or whole list.
    GARCH_single_fc(self, n_steps, ret_vec)
        computes n_steps days forward estimation, returns the last price or the list of prices.
    montecarlo_sim(self, n_reps, n_steps)
        performs montecarlo simulation of the price of the asset with n_reps repetitions.
    hist_simulation(self, std)
        estimates fitted model over the period and returns residuals.
    pdf_func(self, x, steps, r, up_lim, low_lim, prec)
        returns the probability mass point of the model at x (log-price) at a given point in the future (nsteps)
    cdf_func(self, x, steps, r, up_lim, low_lim, prec)
        returns the cumulative probability of the log-price being less or eqaul than x at a given point in the future (nsteps)

    '''
    def __init__(self, timeseries=None, r_f=0.):
        self.timeseries = list(timeseries)
        self.omega = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.p_lambda = None
        self.lr_var = None
        self.r_f = r_f
        self.gamma_star = None
        self.h_t0 = None
        self.std_errors = None
     
    # class method that estimates the parameters of the model    
    def GARCH_fit(self, symmetric=False):
        '''

        Parameters
        ----------
        symmetric : bool, optional
            if True the GARCH model is symmetric, the gamma parameter is held 0. The default False.
        
        Returns
        -------
        None.

        '''
        r = self.r_f
        ts = self.timeseries.copy()
        
        r_t = [log(ts[i]/ts[i-1]) for i in range(1,len(ts))]
        
        omega = variance(r_t)
        alpha = 0.1 * variance(r_t)
        beta = 0.1
        gam = 0.
        lam = -0.5

        omega = -log((1-omega)/omega)
        alpha = -log((1-alpha)/alpha)
        beta = -log((1-beta)/beta)    
    
        par = [omega, alpha, beta, gam, lam]
        res = minimize(llhngarch, par, args=(r_t, r, False), method='L-BFGS-B')

        # computing standard errors of garch model through the inverse of the hessian matrix
        hess_inv = res.hess_inv.todense()
        diag = np.diag(hess_inv)

        self.std_errors = np.sqrt(diag)

        omega = (1/(1 + exp(-res.x[0])))
        alpha = (1/(1 + exp(-res.x[1])))
        beta = (1/(1 + exp(-res.x[2])))
        lam = res.x[4]
        if symmetric: gam =0.
        else: gam = res.x[3]
        
        self.gamma_star = lam + gam + 0.5
    
        self.omega, self.alpha, self.beta, self.gamma, self.p_lambda = omega, alpha, beta, gam, lam
        
        
        
        print('Estimation results:')
        print('-------------------')
        print('Omega:    ' + str(omega))
        print('Alpha:    ' + str(alpha))
        print('Beta:     '+str(beta))
        if not symmetric: print('Gamma:    ' + str(gam))
        print('Lambda:   ' + str(lam))
        
        persistence = beta + alpha * pow(gam,2)
        print('\n')
        print('Model persistence: ' + str(round(persistence,6)))
        self.lr_var = (omega + alpha)/(1-persistence)
        print('Long-run variance: ' + str(round(self.lr_var,6)))
        print('\n')
        
    # class method that returns the long-run variance of the GARCH model    
    def lr_variance(self):
        '''

        Returns
        -------
        float
            long-run variance of the estimated process.

        '''
        print('Model variance')
        print('----------------')
        print('Daily variance: ' + str(round(self.lr_var*100,4)) + '%')
        return self.lr_var
    
    # class method that returns the long-run volatility of the GARCH model (sqrt of the variance)
    def lr_vol(self):
        '''

        Returns
        -------
        float
            long-run volatility of the estimated process.

        '''
        print('Model volatility')
        print('----------------')
        print('Annualized volatility: ' + str(round(sqrt(self.lr_var)*sqrt(252)*100,4)) + '%')
        print('Daily volatility:      ' + str(round(sqrt(self.lr_var)*100,4)) + '%')
        return sqrt(self.lr_var)

    def get_std_errors(self):
        '''

        Returns
        -------
        array
            standard errors of the estimates computed using inverse hessian matrix.

        '''
        print('Standard errors for the estimates:')
        print('----------------------------------')
        print('omega:  ' + str(self.std_errors[0]))
        print('alpha:  ' + str(self.std_errors[1]))
        print('beta:   ' + str(self.std_errors[2]))
        print('gamma:  ' + str(self.std_errors[3]))
        print('lambda: ' + str(self.std_errors[4]))

        return self.std_errors

    # class method that estimates the final variance of the ts of log returns
    def ts_var(self, vec=False):
        '''
        
        Parameters
        ----------
        vec : bool, optional
            if True returns the whole process of the estimated variance. The default is False.

        Returns
        -------
        float or list
            returns the terminal variance of the process, if vec = True returns the whole process of the estimated variance.

        '''
        ts = self.timeseries.copy()
        r_f = self.r_f
        
        params = [self.omega, self.alpha, self.beta, self.gamma, self.p_lambda]
        
        r_t = [log(ts[i]/ts[i-1]) for i in range(1,len(ts))]
        
        h_vec = []
        h_t = variance(r_t)
        h_vec.append(h_t)
        n = len(r_t)
        
        for i in range(1,n+1):
            z_t = pow((r_t[i-1] - r_f - params[4] * h_t - params[3] * h_t), 2)
            h_t = params[0] + params[1] * z_t / h_t + params[2] * h_t
            h_vec.append(h_t)
             
        self.h_t0 = h_vec[-1]    
        
        # before 
        # if vec:
        #     return h_vec
        # changed because gives problems with filtering as we don't have any record for the (final day+1)
        # but we can theoretically compute the variance
        
        if vec:
            return h_vec[:-1]
        else:
            return h_vec[-1]
        
    # function to get standrardized residuals is residuals/sqrt(h_t)
    def hist_simulation(self):
        '''
        

        Returns
        -------
        residuals : list
            returns standardized residuals of the fitted time series.

        '''
        
        # init
        ts = self.timeseries
        r_f = self.r_f
        
        params = [self.omega, self.alpha, self.beta, self.gamma, self.p_lambda]
        
        r_t = [log(ts[i]/ts[i-1]) for i in range(1,len(ts))]
        
        h_vec =[]
        r_vec = []
        
        # h_0 is the sample variance
        h_t = variance(r_t)
        z = (r_t[0] - r_f - params[4] * h_t) / sqrt(h_t)
        
        
        for i in range(1,len(r_t)):
            h_t = params[0] + params[2] * h_t + params[1]*pow(z - params[3]*sqrt(h_t),2)
            h_vec.append(h_t)
            z = (r_t[i] - r_f - params[4] * h_t) / sqrt(h_t)
            r_vec.append(z)        
        
        residuals = [r/sqrt(v) for r,v in zip(r_vec, h_vec)]
        
        return residuals
    
    # can be updated to accomodate montecarlo simulation
    def GARCH_single_fc(self, n_steps=252, vec=False):
        '''

        Parameters
        ----------
        n_steps : int, optional
            number of future periods to forecast. The default is 252 (trading days in a year).
        vec : bool, optional
            if True returns the whole price process. The default is False.

        Returns
        -------
        float or list
            returns the terminal price, if vec = True returns the whole price process.

        '''
        
        params = [self.omega, self.alpha, self.beta, self.gamma_star, self.p_lambda]

        h_t = self.h_t0
        z_star = normal(0,1)
        s_t = log(self.timeseries[-1])
        s = []

        for i in range(n_steps):
            
            # volatility process
            h_t = params[0] + params[2] * h_t + params[1] * pow(z_star - params[3] * sqrt(h_t), 2)
            # logreturns process
            z_star = normal(0,1)
            s_t = s_t + self.r_f - 0.5 * h_t + sqrt(h_t) * z_star

            s.append(exp(s_t))

        if vec:    
            return s
        else:
            return s[-1]
    
    def montecarlo_sim(self, n_reps=5e3, n_steps=252):
        '''

        Parameters
        ----------
        n_reps : int, optional
            number of simulations. The default is 5e3.
        n_steps : int, optional
            number of future periods to forecast. The default is 252 (trading days in a year).

        Returns
        -------
        float
            average terminal value of the asset.

        '''
        res = []
        
        for i in np.arange(n_reps):
            s = HNGarch.GARCH_single_fc(self, n_steps, vec=False)
            res.append(s)
        
        return sum(res)/n_reps

########################
#  Heston-Nandi GARCH  #
#   Num. Integration   #
########################

    def pdf_func(self, x, steps, r=0., up_lim=100, low_lim=0, prec=10000):
        '''

        Parameters
        ----------
        x : float/array
            point/s of evaluation of the probability density function.
        steps : int
            number of days to expiration.
        r : float, optional
            risk-free interest rate. The default is 0..
        up_lim : float, optional
            superior bound of the integration interval. The default is 100.
        low_lim : TYPE, optional
            inferior bound of the integration interval. The default is 0.
        prec : int, optional
            number of points for integral computation. The default is 10000.

        Returns
        -------
        float/array
            value of the probability density function in x, or array of pdfs (if x is an array).

        '''
        # init
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        gam = self.gamma_star
        lam = -0.5
        
        s = self.timeseries[-1]
        h = self.ts_var(vec=False)
        
        hi = float(up_lim-low_lim)/prec
        t = np.linspace(low_lim + hi/2, up_lim- hi/2, prec)
        t[t == 0] = 1e-10
        
        a = np.zeros(len(t))
        b = np.zeros(len(t))
    
        it = t * np.complex(0,1)
        for i in range(steps):
            a = A_loop(a, b, r, it, omega, alpha)
            b = B_loop(b, it, alpha, beta, gam, lam)
        
        vec = x.reshape(-1,1)
        
        f = s**it
        g = np.exp(a + b * h)
        func = np.real(np.exp(-it*vec) * f * g)
        
        area = func*hi
        return area.sum(axis=1)/np.pi
    
    def cdf_func(self, x, steps, r=0., up_lim=100, low_lim=0, prec=10000):
        '''

        Parameters
        ----------
        x : float/array
            point/s of evaluation of the cumulative density function.
        steps : int
            number of days to expiration.
        r : float, optional
            risk-free interest rate. The default is 0..
        up_lim : float, optional
            superior bound of the integration interval. The default is 100.
        low_lim : TYPE, optional
            inferior bound of the integration interval. The default is 0.
        prec : int, optional
            number of points for integral computation. The default is 10000.

        Returns
        -------
        float/array
            value of the cumulative density function in x, or array of cdfs (if x is an array).

        '''
        # init
        omega = self.omega
        alpha = self.alpha
        beta = self.beta
        gam = self.gamma_star
        lam = -0.5
        
        s = self.timeseries[-1]
        h = self.ts_var(vec=False)
        
        hi = float(up_lim-low_lim)/prec
        t = np.linspace(low_lim + hi/2, up_lim- hi/2, prec)
        t[t == 0] = 1e-10
        
        a = np.zeros(len(t))
        b = np.zeros(len(t))
    
        it = t * np.complex(0,1)
        for i in range(steps):
            a = A_loop(a, b, r, it, omega, alpha)
            b = B_loop(b, it, alpha, beta, gam, lam)
        
        vec = x.reshape(-1,1)
        
        f = s**it
        g = np.exp(a + b * h)
        j = np.exp(-it*vec) * f * g
        func = np.real(j/it)
        
        area = func*hi
        return 0.5 - area.sum(axis=1)/np.pi
    
#%% MAIN
if __name__ == '__main__':
    import time
    import pandas as pd
    from matplotlib import pyplot as plt

    # importing timeseries as list of prices
    ts = pd.read_csv(r'C:\Users\edoardo_berton\Desktop\copula_op\code\WTI_ts.csv')['Close']

    # init of the class
    model = HNGarch(ts)

    # fitting the model
    model.GARCH_fit()

    vec1 = model.hist_simulation()



    x = np.arange(-5, np.log(7000), 0.01)
    x[x==0] = 1e-10

    pdf = model.pdf_func(x, 90)
    cdf = model.cdf_func(x, 90)


#%%
    std_errs = model.get_std_errors()

    model.lr_vol()

    # vector of variances of the estimated model
    vec = model.ts_var(vec=True)

    # # plot the simulated garch model over a 252d future interval
    j = pd.DataFrame()
    # how many simulations to plot
    number_scenarios = 10
    for i in range(number_scenarios):
        y = pd.DataFrame(model.GARCH_single_fc(252,True))
        j = pd.concat([j,y], axis=1)

    j.plot(legend=False)

    # montecarlo simulation on the price of the underlying (num_simulations, days period)
    avg_price = model.montecarlo_sim(10000, 252)
