"""
Main Driver Code

"""

import ML.train
import matplotlib.pyplot as plt

if __name__ == "__main__":
    F, R, lossG, lossD = ML.train.train_automate(5)
    plt.plot(R, label="Real")
    plt.plot(F, label="Fake")
    plt.legend()
    plt.show()
    plt.plot(lossG, label="Gen LOSS")
    plt.plot(lossD, label="Fake LOSS")
    plt.legend()
    plt.show()
