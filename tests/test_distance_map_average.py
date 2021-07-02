from minisom import MiniSom
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt


def test_distance_map():
    """
    Compare distance maps with averaged or added neighbour distances.
    """
    iris = load_iris()
    X = iris["data"]
    som = MiniSom(10, 10, X.shape[1])
    som.train(X, 10000, verbose=True)
    # map with added distances
    dist_map = som.distance_map()
    # map with averaged distances
    dist_map_average = som.distance_map(neighbour_average=True)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dist_map)
    ax[0].set_title("Default map")
    ax[1].imshow(dist_map_average)
    ax[1].set_title("Averaged map")
    plt.show()
