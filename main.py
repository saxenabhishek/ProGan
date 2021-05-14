"""
Main Driver Code

"""
from ML.proTrain import train

import torch

torch.autograd.set_detect_anomaly(True)


def main():
    gan = train(
        "Data",
        128, #batch size
        [90, 10, 0], #data split
        "ModelWeights", #save location
        lr=[0.001, 0.001], #learning Rates
        merge_samples_Const=17, # merge rate
        loadmodel=True, # continue training
        PlotInNotebook=False, # if in notebook
    )
    gan.step_up()  # -> 8,8
    gan.trainer(17, 100)
    gan.plot_trainer()
    gan.trainer(17, 100)
    gan.trainer(17, 100)
    gan.step_up()  # -> 16,16
    gan.trainer(17, 100)
    gan.trainer(17, 100)
    gan.step_up()  # -> 32,32
    gan.trainer(17, 100)
    gan.trainer(17, 100)
    gan.step_up()  # -> 64,64
    gan.trainer(17, 100)
    gan.trainer(17, 100)
    gan.step_up()  # -> 128,128
    gan.trainer(17, 100)
    gan.trainer(17, 100)

if __name__ == "__main__":
    main()
