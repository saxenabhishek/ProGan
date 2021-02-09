"""
Main Driver Code

"""

import ML.train

import matplotlib.pyplot as plt


def main(e, path="data", split=[20, 1, 0], vec_shape=1000, batch_size=25):
    F, R, lossG, lossD = ML.train.train_automate(
        e, path, split, vec_shape=vec_shape, batch_size=batch_size
    )
    plt.plot(R, label="Real")
    plt.plot(F, label="Fake")
    plt.legend()
    plt.show()
    plt.plot(lossG, label="Gen LOSS")
    plt.plot(lossD, label="Fake LOSS")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(5)
