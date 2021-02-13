"""
Main Driver Code

"""

from ML.train import train
import matplotlib.pyplot as plt
from time import time


def main():
    modeltrain = train(
        path="Data", epochs=3, batch_size=10, vec_shape=100, split=[1, 20, 0], noisedim=100, display_step=10000
    )
    starting = time()
    modeltrain.trainer()
    print(time() - starting)
    modeltrain.plot_trainer()


if __name__ == "__main__":
    main()
