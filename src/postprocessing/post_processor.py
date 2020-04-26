from scipy.spatial import distance
from matplotlib import cm as c
import numpy as np
import matplotlib.pyplot as plt


class PostProcessor:
    def __init__(self):
        self._dist_matrix = None

    def find_min_distance(self, centers, plot=True):
        '''
        return min eculidian distance between predict anchor boxes
        '''
        self._dist_matrix = distance.cdist(centers, centers)
        np.fill_diagonal(self._dist_matrix, np.nan)
        return np.nanmin(self._dist_matrix)

    def plot_centers_distrobution(self, centers):
        scatter_cents = np.array(
            centers).reshape(-1, 2).astype('float32')
        plt.scatter(x=scatter_cents[:, 0], y=scatter_cents[:, 1])
        plt.show()
