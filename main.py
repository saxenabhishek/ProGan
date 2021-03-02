"""
Main Driver Code

"""

from ML.train import train
import matplotlib.pyplot as plt
from time import time


def main(path, epochs, split=[1, 2000, 0], display_step=250):
    modeltrain = train(
        path=path, epochs=epochs, batch_size=4, vec_shape=100, split=split, noisedim=100, display_step=display_step
    )
    starting = time()
    modeltrain.trainer()
    print(time() - starting)
    modeltrain.plot_trainer()


if __name__ == "__main__":
    main()
