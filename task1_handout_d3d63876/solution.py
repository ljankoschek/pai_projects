import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import KMeans

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


def undersample(
        train_coordinates: np.ndarray,
        train_area_flags: np.ndarray,
        train_targets: np.ndarray,
        n: int
) -> (np.ndarray, np.ndarray):
    random_state = random.randint(0, 4294967295)

    # Perform stratified sampling with the random random_state
    train_coordinates = train_test_split(train_coordinates, test_size=len(train_coordinates) - n,
                                         stratify=train_area_flags, random_state=random_state)[0]

    train_targets = train_test_split(train_targets, test_size=len(train_targets) - n,
                                     stratify=train_area_flags, random_state=random_state)[0]


    return train_coordinates, train_targets

def stratified_cluster_undersample(train_coordinates: np.ndarray,
        train_area_flags: np.ndarray, train_targets: np.ndarray, samples: int):
    clusters = 15
    train_coordinates_special = train_coordinates[train_area_flags == 1.0]
    train_targets_special = train_targets[train_area_flags == 1.0]
    train_coordinates_notSpecial = train_coordinates[train_area_flags == 0.0]
    train_targets_notSpecial = train_targets[train_area_flags == 0.0]
    specialSize = train_coordinates_special.size / train_coordinates.size
    notSpecialSize = train_coordinates_notSpecial.size / train_coordinates.size
    kmeansSpecial = KMeans(n_clusters=15, random_state=0, n_init="auto")
    clusters_special = kmeansSpecial.fit(train_coordinates_special)
    labels = kmeansSpecial.labels_



    random_state1 = random.randint(0, 4294967295)
    random_state2 = random.randint(0, 4294967295)

    train_coordinates1 = train_test_split(train_coordinates_notSpecial, test_size=int(len(train_coordinates_notSpecial) - (notSpecialSize*samples)),
                                           random_state=random_state1)[0]
    train_targets1 = train_test_split(train_targets_notSpecial,
                                          test_size=int(len(train_targets_notSpecial) - (notSpecialSize * samples)),
                                           random_state=random_state1)[0]

    train_coordinates2 = train_test_split(train_targets_special, test_size= int(len(train_targets_special) - (specialSize*samples)),
                                     stratify=labels, random_state=random_state2)[0]
    train_targets2 = train_test_split(train_targets_special, test_size= int(len(train_targets_special) - (specialSize*samples)),
                                     stratify=labels, random_state=random_state2)[0]

    train_coordinates = np.stack((train_coordinates1,train_coordinates2),axis=1)
    train_targets = np.stack((train_targets1,train_targets2),axis=1)

    return train_coordinates, train_targets

    #sortedData = [[] for i in range(clusters)]
    #for i in range(clusters):
     #   for j in range(labels.size):
      #      if labels[j] == i:
      #          sortedData[i].append(train_coordinates_special[j])

   # testData = [[] for i in range(clusters)]
   # for i in range(clusters):
    #    for j in range(samples):


    print(labels)

class Model(object):
    """
    Model for this task.
    You need to implement the train_model and generate_predictions methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        # OLD CODE
        #self.rng = np.random.default_rng(seed=0)
        #self.perform_undersampling = True
        #self.undersampling_size = 5000
        #self.kernel = Matern(length_scale=15.0, nu=1.5, length_scale_bounds="fixed")
        #self.gp = GaussianProcessRegressor(self.kernel,n_restarts_optimizer=50)

        # TODO: Add custom initialization for your model here if necessary
        # we use Approximation of GP with multiple local GPs for the candidate residential areas
        # for the residential areas we use a global gp
        self.global_kernel = Matern(length_scale=21.0, nu=1.5, length_scale_bounds="fixed") # kernel for residential areas
        self.global_gp = GaussianProcessRegressor(kernel=self.global_kernel, n_restarts_optimizer=10) # global gp for residential areas

        self.num_clusters = 14 # numbers of clusters (in candidate residentiala areas)
        self.cluster_centers = None  # store the cluster centers
        self.local_gps = []  # store local GP models for each cluster
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto") # init KMeans clustering model
        

    def generate_predictions(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # OLD CODE
        #z_q = norm.ppf(0.95)
        #predictions = np.ndarray([])
        #predictions = gp_mean
        #for i in range(gp_mean.size):
        #    if test_area_flags[i] == 1.0:
        #       predictions[i] = gp_mean[i] + 50000*(z_q * gp_std[i])
        

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        # keep track of original indices so we can put the predictions back together in the correct order
        original_indices = np.arange(len(test_coordinates))

        ## TODO: Use the GP posterior to form your predictions here        
        # create return ndarrays
        predictions = np.zeros(len(test_coordinates))
        gp_means = np.zeros(len(test_coordinates))
        gp_stds = np.zeros(len(test_coordinates))

        # split test coordinates into residential areas and candidate residential areas (area flag = 1.0)
        mask = test_area_flags == 1.0
        candidate_test_indices = original_indices[mask]
        residential_areas_test_indices = original_indices[~mask]
        candidate_test_coordinates = test_coordinates[candidate_test_indices]
        residential_areas_test_coordinates = test_coordinates[residential_areas_test_indices]
        
        # use cKDTree structure to find nearest cluster for each candidate test point
        # https://ceur-ws.org/Vol-2786/Paper28.pdf
        tree = cKDTree(self.cluster_centers)
        _, nearest_clusters = tree.query(candidate_test_coordinates)

        # predict test points for each cluster (candidate residential areas) with local gp
        for cluster_idx in range(self.num_clusters):
            # identify points in cluster and get indices
            points_in_cluster = candidate_test_coordinates[nearest_clusters == cluster_idx]
            points_indices_in_cluster = candidate_test_indices[nearest_clusters == cluster_idx]
            
            gp_model = self.local_gps[cluster_idx]
            gp_mean, gp_std = gp_model.predict(points_in_cluster, return_std=True)
            gp_means[points_indices_in_cluster] = gp_mean
            gp_stds[points_indices_in_cluster] = gp_std

            # asymmetric cost
            # compute modified prediction to avoid underestimating the pollution concentration in the candidate residential areas
            z_q = norm.ppf(0.95)
            predictions[points_indices_in_cluster] = gp_mean + 68000 * (z_q * gp_std)
            
        # predict for residential area coordinates using global gp
        gp_mean, gp_std = self.global_gp.predict(residential_areas_test_coordinates, return_std=True)
        predictions[residential_areas_test_indices] = gp_mean
        gp_means[residential_areas_test_indices] = gp_mean
        gp_stds[residential_areas_test_indices] = gp_std

        return predictions, gp_means, gp_stds


    def train_model(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_area_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        # OLD CODE
        # if self.perform_undersampling:
        #    train_coordinates, train_targets = stratified_cluster_undersample(train_coordinates, train_area_flags, train_targets, 1000)
        #
        #if self.perform_undersampling:
        #    train_coordinates, train_targets = undersample(
        #        train_coordinates,
        #        train_area_flags,
        #        train_targets,
        #        self.undersampling_size
        #    )
        #
        #self.gp.fit(train_coordinates,train_targets)
        #print("Log marginal likelyhood for kernel:")
        #print(self.gp.log_marginal_likelihood(theta=self.gp.kernel_.theta))
        #print("Parameters")
        #print(self.gp.kernel_.get_params())
        
        # split train coordinates into residential areas and candidate residential areas (area flag = 1.0)
        mask = train_area_flags == 1.0
        candidate_train_coordinates = train_coordinates[mask]
        candidate_train_targets = train_targets[mask]

        residential_area_train_coordinates = train_coordinates[~mask]
        residential_area_train_targets = train_targets[~mask]

        # fit KMeans clustering and store cluster centers
        self.kmeans.fit(candidate_train_coordinates)
        self.cluster_centers = self.kmeans.cluster_centers_

        for cluster_idx in range(self.num_clusters):
            # get points and targets of current cluster
            cluster_indices = np.where(self.kmeans.labels_ == cluster_idx)[0]
            cluster_points = candidate_train_coordinates[cluster_indices]
            cluster_targets = candidate_train_targets[cluster_indices]

            # train gp model for current cluster
            kernel = Matern(length_scale=13.0, nu=1.5, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            gp.fit(cluster_points, cluster_targets)
            self.local_gps.append(gp)
        
        # train global gp
        self.global_gp.fit(residential_area_train_coordinates, residential_area_train_targets)


# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function 
def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags


# You don't have to change this function
def execute_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.generate_predictions(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """



    train_coordinates = train_x[:, :2]
    print(train_coordinates)
    train_area_flags = train_x[:, -1].astype(bool)
    test_coordinates = test_x[:, :2]
    test_area_flags = test_x[:, -1].astype(bool)

    assert train_coordinates.shape[0] == train_area_flags.shape[0] and test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1 and test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = extract_area_information(train_x, test_x)
    
    # Fit the model
    print('Training model')
    model = Model()
    model.train_model(train_y, train_coordinates, train_area_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.generate_predictions(test_coordinates, test_area_flags)
    print(predictions)


    if EXTENDED_EVALUATION:
        execute_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()


