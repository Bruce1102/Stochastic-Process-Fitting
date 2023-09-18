import numpy as np
import math
from scipy.stats import ncx2
from scipy.optimize import minimize
import statsmodels.api as sm

class GBM:
    def __init__(self, initial_price: float=100, mu: float=0.05, sigma: float=0.1, tau: float=1, 
                 dividend: float=0, n_steps:int = 1_000):
        """ Initialize the StochasticProcessSimulation.

        Parameters:
        - initial_price : Initial asset price.
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        self.initial_price     = initial_price
        self.discrete_dividend = dividend * tau / n_steps
        self.mu                = mu - self.discrete_dividend
        self.sigma             = sigma
        self.tau               = tau
        self.dt                = tau / n_steps
        self.n_steps           = n_steps
        self.simulation = np.zeros(n_steps)

    def set_params(self, params: tuple):
        """Setting given parameters to itself"""
        self.mu = params[0]
        self.sigma = params[1]


    def _moments(self, data: np.array) -> tuple:
        """Estimate GBM parameters using Method of Moments."""
        changes = (data[1:] - data[:-1]) / data[:-1]
        mu = np.mean(changes)
        sigma = np.std(changes)
        
        # Compute parameters
        params = (mu, sigma)
        
        # Set existing model to newly computed parameters
        self.set_params(params)

        return params

    def _least_squares(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Least Squares Method."""

        def _ls_error(params: tuple, data: np.array) -> float:
            """Least squares error for the Vasicek model."""
            mu, sigma = params
            predicted = data[:-1] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=len(data)-1))
            return np.sum((data[1:] - predicted)**2)

        init_params = [0.05, 0.02]
        result = minimize(_ls_error, init_params, args=(data), method='L-BFGS-B')
        params = result.x

        self.set_params(params)

        return params

    def _mle(self, data: np.array) -> tuple:
        """Estimate GBM parameters using Maximum Likelihood Estimation."""
        def _log_likelihood(params: tuple, data: np.array) -> float:
            mu, sigma = params
            n = len(data) - 1
            log_returns = np.log(data[1:] / data[:-1])
            
            # Compute the log-likelihood
            log_likelihood = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sigma**2 * self.dt) - 0.5 * np.sum((log_returns - mu*self.dt)**2) / (sigma**2 * self.dt)
            
            return -log_likelihood

        
        init_params = [0.05, 0.02]
        result = minimize(_log_likelihood, init_params, args=(data), bounds = [(None, None), (0.00000001, 0.5)])
        params = result.x

        self.set_params(params)

        return params



    def fit(self, data: np.array, method: str='moments') -> tuple:
        if method not in ['moments', 'least square', 'mle']:
            raise ValueError("Invalid fitting method type. Choose from 'moments', 'least square' or 'mle'.")

        if method == 'moments':
            return self._moments(data)
        elif method == 'least square':
            return self._least_squares(data)
        elif method == 'mle':
            return self._mle(data)


    def _drift(self, price:float) -> float:
        """Compute the drift term for the given price."""
        return self.mu * price * self.dt

    def _diffusion(self, price:float) -> float:
        """Compute the diffusion term for the given price."""
        return price * self.sigma * np.sqrt(self.dt) * np.random.normal()

    def drift_diffusion(self, price:float) -> float:
        """Compute the combined drift and diffusion for the given price."""
        return self._drift(price) + self._diffusion(price)

    def simulate(self):
        """Simulate the price dynamics using the drift-diffusion model."""
        self.simulation[0] = self.initial_price
        for i in range(1, self.n_steps):
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1])

    def get_simulation(self) -> np.array:
        """Return the simulated price dynamics."""
        return self.simulation


class VasicekProcess(GBM): 
    def __init__(self, initial_price:float=100, kappa: float=0.1, theta: float=0.05, sigma:float=0.2, 
                 tau:float=1, dividend: float=0, n_steps:int = 1_000):
        """ Initialize the Vasicek Process.

        Parameters:
        - initial_price : Initial asset price.
        - kappa         : Speed of mean reversion
        - theta         : Amplitude of mean reversion
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        super().__init__(initial_price, 0, sigma, tau, dividend, n_steps)
        self.kappa  = kappa
        self.theta  = theta
        self.bounds = [(0.00001, 5)]
    
    def set_params(self, params: tuple):
        kappa, theta, sigma = params

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def _moments(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Method of Moments."""
        mu = np.mean(data)
        sigma2 = np.var(data)
        rho = np.corrcoef(data[:-1], data[1:])[0,1]
        
        # Compute parameters
        kappa = -np.log(rho)
        theta = (mu * (1 - rho))
        sigma = np.sqrt(sigma2 * 2 * kappa / (1 - rho**2))
        params = (kappa, theta, sigma)
        
        # Set existing model to newly computed parameters
        self.set_params(params)

        return params

    def _least_squares(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Least Squares Method."""

        def _ls_error(params: tuple, data: np.array, dt: float=1.0) -> float:
            """Internal function for computing the least squares error for the Vasicek model."""
            kappa, theta, sigma, = params
            r0 = data[0]
            predicted = r0 + kappa * (theta - data[:-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=len(data)-1)
            return np.sum((data[1:] - predicted)**2)

        init_params = [0.1, 0.05, 0.02]
        result = minimize(_ls_error, init_params, args=(data))
        params = result.x

        self.set_params(params)

        return params

    def _mle(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Maximum Likelihood Estimation."""

        def _log_likelihood(params: tuple, data: np.array, dt: float=1.0) -> float:
            """Internal function for computing the negative log-likelihood for the Vasicek model."""
            kappa, theta, sigma = params
            n = len(data) - 1
            
            # Compute the log-likelihood (separated for readability)
            log_likelihood = (-0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
                              - np.sum((data[1:] - data[:-1] - kappa * (theta - data[:-1]) * dt)**2) / (2 * sigma**2 * dt))
            
            return -log_likelihood
        
        init_params = [0.1, 0.05, 0.02]
        result = minimize(_log_likelihood, init_params, args=(data), method='L-BFGS-B')
        return result.x

    def _drift(self, price:float) -> float:
        """Compute the drift term for the given price."""
        return self.kappa * (self.theta - price ) * self.dt

    def _diffusion(self, price: float) -> float:
        """Compute the diffusion term for the given price."""
        return self.sigma * np.sqrt(self.dt) * np.random.normal()

class CIRProcess(GBM): 
    def __init__(self, initial_price:float=100, kappa: float=0.1, theta: float=0.05, sigma:float=0.2, 
                 tau:float=1, dividend: float=0, n_steps:int = 1_000):
        """ Initialize the Cox-Ingersoll-Ross Process.

        Parameters:
        - initial_price : Initial asset price.
        - kappa         : Speed of mean reversion
        - theta         : Amplitude of mean reversion
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        super().__init__(initial_price, 0, sigma, tau, dividend, n_steps)
        self.kappa  = kappa
        self.theta  = theta
    
    def set_params(self, params: tuple):
        kappa, theta, sigma = params

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def _moments(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Method of Moments."""
        mu = np.mean(data)
        delta_data = data[1:] - data[:-1]
        sigma2 = np.var(delta_data) 
        rho = np.corrcoef(data[:-1], data[1:])[0,1]
        
        # Compute parameters
        kappa = -np.log(rho)
        theta = (mu * (1 - rho))
        # sigma = np.sqrt(sigma2 * 2 * kappa / (1 - rho**2))
        sigma = np.sqrt(sigma2 / (np.mean(data[:-1]) * self.dt))
        params = (kappa, theta, sigma)
        
        # Set existing model to newly computed parameters
        self.set_params(params)

        return params

    def _least_squares(self, data: np.array) -> tuple:
        """Estimate CIR parameters using Least Squares Method."""

        def _ls_error(params: tuple, data: np.array, dt: float=1.0) -> float:
            """Internal function for computing the least squares error for the CIR model."""
            kappa, theta, sigma = params
            predicted = data[:-1] + kappa * (theta - data[:-1]) * dt + sigma * np.sqrt(data[:-1] * dt) * np.random.normal(size=len(data)-1)
            return np.sum((data[1:] - predicted)**2)

        init_params = [0.1, 0.05, 0.2]
        result = minimize(_ls_error, init_params, args=(data), bounds = [(None, None), (None, None), (0.000000001, 1)], method = 'Nelder-Mead')
        params = result.x

        self.set_params(params)

        return params

    def _mle(self, data: np.array) -> tuple:
        """Estimate Vasicek parameters using Maximum Likelihood Estimation."""

        def _log_likelihood(params: tuple, data: np.array, dt: float=1.0) -> float:
            """Internal function for computing the negative log-likelihood for the Vasicek model."""
            kappa, theta, sigma = params
            n = len(data) - 1
            
            # Compute the log-likelihood (separated for readability)
            log_likelihood = (-0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
                              - np.sum((data[1:] - data[:-1] - kappa * (theta - data[:-1]) * dt)**2) / (2 * sigma**2 * dt))
            
            return -log_likelihood
        
        init_params = [0.1, 0.05, 0.2]
        result = minimize(_log_likelihood, init_params, args=(data), bounds = [(None, None), (None, None), (0.000000001, 1)])
        return result.x

    def _drift(self, price:float) -> float:
        """Compute the drift term for the given price."""
        return self.kappa * (self.theta - price ) * self.dt

    def _diffusion(self, price: float) -> float:
        """Compute the diffusion term for the given price."""
        return self.sigma * np.sqrt(price) * np.sqrt(self.dt) * np.random.normal()


class StochasticVolatility(GBM):
    def __init__(self, initial_price:float, volatility_process:VasicekProcess, mu: float=0.01, dividend: int=0, tau:float=1, n_steps:int = 1_000):
        """
        Initialize the Stochastic Volatility model.

        Parameters:
        - initial_price      : Initial asset price.
        - mu                 : Drift term for the asset.
        - volatility_process : CIR Process modeling the stochastic volatility (Must be pre-fitted).
        - tau                : Duration of simulation (in years).
        - n_steps            : Number of time steps to simulate.
        """

        if volatility_process.tau != tau:
            raise ValueError("Mismatched tau")
        if volatility_process.n_steps != n_steps:
            raise ValueError("Mismatched n_steps")

        super().__init__(initial_price, mu, None, tau, dividend, n_steps) # Will use our own sigma process

        
        self.volatility_process = volatility_process

        volatility_process.simulate()
        self.volatilities = volatility_process.get_simulation()

    def set_params(self, param):
        """Setting given parameters to itself"""
        self.mu = param


    def _moments(self, data: np.array) -> tuple:
        """Estimate heston model parameters using Method of Moments."""
        # Compute changes
        changes = (data[1:] - data[:-1]) / data[:-1]
        mu = np.mean(changes)

        # Set existing model to newly computed parameters
        self.set_params(mu)

        return mu

    def _least_squares(self, data: np.array) -> float:
        """Estimate Heston parameters using Least Squares Method."""

        self.volatility_process.simulate()
        volatilities = self.volatility_process.get_simulation()

        def _ls_error(params: tuple, data: np.array) -> float:
            mu = params[0]
            # Simulate Heston model with given mu and pre-fitted volatility process
            predicted = data[:-1] * np.exp((mu - 0.5 * volatilities[:-1]) * self.dt + np.sqrt(volatilities[:-1] * self.dt) * np.random.normal(size=len(data)-1))
            return np.sum((data[1:] - predicted)**2)

        init_params = [self.mu]
        result = minimize(_ls_error, init_params, args=(data))
        params = result.x

        self.set_params(params)

        return params

    def _mle(self, data: np.array) -> float:
        pass


    def fit(self, data: np.array, method = 'moments', process: str='heston') -> tuple:

        if method not in ['moments', 'least square', 'mle']:
            raise ValueError("Invalid fitting method type. Choose from 'moments', 'least square' or 'mle'.")
        if process not in ['heston', 'volatility']:
            raise ValueError("Invalid process to fit, choose from 'heston', 'volatility'.")

        if process == 'volatility':
            self.volatilities.fit(data, method)
        else:
            super().fit(data, method)

    def _volatility_diffusion(self, price, volatility):
        return price * np.sqrt(volatility) * np.sqrt(self.dt) * np.random.normal()

    def drift_diffusion(self, price:float, volatility:float) -> float:
        """Compute the combined drift and diffusion for the given price."""
        drift = self._drift(price)
        diffusion = self._volatility_diffusion(price, volatility)
        return drift + diffusion

    def simulate(self):
        """Simulate the price dynamics using the drift-diffusion model."""
        # Simulating volatilties first
        self.volatility_process.simulate()
        self.simulation[0] = self.initial_price

        volatilities = self.volatility_process.get_simulation()
        for i in range(1, self.n_steps):
            # Update stock price using the drift-diffusion model
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1], volatilities[i-1])