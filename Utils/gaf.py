import numpy as np


def minmax(x):
    """
    x: 1D np array
    """
    m = x.min()
    M = x.max()

    x_ = (2 * x - m - M) / (M - m)
    return x_


def GAF(x):
    """
    x: 1D np array
    """
    xv, xv_ = np.meshgrid(x, x)
    G = xv * xv_ - np.sqrt(1 - xv ** 2) * np.sqrt(1 - xv_ ** 2)
    return G


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    X = np.random.randn(30)
    X_ = minmax(X)
    G = GAF(X_)

    plt.imshow(G)
    plt.show()
