"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process.kernels import Matern, DotProduct, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        std_f = 0.15
        std_v = 0.0001
        kernel_f = Matern(nu=2.5, length_scale=1, length_scale_bounds="fixed") #Matern with nu = 2.5 or RBF with variance 0.5, lengthscale = 10, 1 or 0.5
        kernel_v = ConstantKernel(4) + DotProduct() + Matern(nu=2.5, length_scale=1, length_scale_bounds="fixed")

        self.gp_f = GaussianProcessRegressor(kernel=kernel_f, alpha=std_f**2, normalize_y=True, optimizer=None)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v, alpha=std_v**2, normalize_y=True, optimizer=None)

        self.x = []
        self.f = []
        self.v = []


    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        x_next = self.optimize_acquisition_function()
        #return np.array([[x_next]]) #np.array version
        return x_next # hier steht float aber unten wird np.array asserted

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)

        # from https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
        # Try Probability of Improvement
        prob = 1 - norm.cdf((mean_v - SAFETY_THRESHOLD) / std_v)
        mean_f = mean_f * prob 

        # use UCB for for v
        beta_v = 5
        ucb_v = mean_v + beta_v * std_v
        
        # use lagrangian relaxation with ucb_v
        #if ucb_v is bigger than threshold, use this as penalty otherwise 0 (= no penalty) 
        # https://arxiv.org/pdf/1906.09459
        l = 2  
        return mean_f - l * max(ucb_v - SAFETY_THRESHOLD, 0)


    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.x.append(x)
        self.f.append(f)
        self.v.append(v)

        x_arr = np.array(self.x).reshape(-1, 1)
        f_arr = np.array(self.f).reshape(-1, 1)
        v_arr = np.array(self.v).reshape(-1, 1)

        self.gp_f.fit(x_arr, f_arr)
        self.gp_v.fit(x_arr, v_arr)


    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        x_arr = np.array(self.x).reshape(-1, 1)
        v_arr = np.array(self.v).reshape(-1, 1)
        valid_x = x_arr[v_arr < SAFETY_THRESHOLD]
        
        if valid_x.size == 0:
            return None
        
        means_f = self.gp_f.predict(x_arr.reshape(-1, 1))
        x_opt = x_arr[np.argmax(means_f)]
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
       # assert x.shape == (1, DOMAIN.shape[0]), \
       #     f"The function recommend_next must return a numpy array of " \
       #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        #obj_val = f(x) + np.randn()
        #cost_val = v(x) + np.randn()
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
