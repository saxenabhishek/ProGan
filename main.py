"""
Main Driver Code

"""

from ML.train import train
import matplotlib.pyplot as plt
from time import time


def main():
    modeltrain = train(
        path="Data", epochs=1, batch_size=4, vec_shape=100, split=[1, 2000, 0], noisedim=100, display_step=1
    )
    starting = time()
    modeltrain.trainer()
    print(time() - starting)
    modeltrain.plot_trainer()


if __name__ == "__main__":
    main()
