"""
Main Driver Code

"""

from ML.train import train 

import matplotlib.pyplot as plt


def main():
    modeltrain = train(path='../fashiondata/img', epochs=1, batch_size=100, vec_shape=100)
    modeltrain.trainer()

if __name__ == "__main__":
    main()
