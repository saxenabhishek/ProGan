"""
Main Driver Code

"""

import ML.train
import matplotlib.pyplot as plt

if __name__ == "__main__":
    R, F = ML.train.train_automate(1)
    plt.plot(R)
    plt.plot(F)
    plt.show()
