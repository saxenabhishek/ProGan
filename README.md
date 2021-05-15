# ✨ ProGAN


<div align="center">
<img align = "center" src="assets\progan-ep41.gif">


##### Training 40 epochs on the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html) dataset
  
###### [top-bottom] 3 numbers on top indicates `Epochs`, `Depth`, and `Alpha-value`, respectively. The first nine image blocks are fake-generated data; the rest are actual samples.

</div>
<br>


## 💡 Project Description

This project is a reimplementation of the paper [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)  by Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen. Check out this [article](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2) for a recap on how GANs work and how progressive gans are different.

This repository is developed like a python package, so it is easy to quickly start training on any dataset. Everything you need is an import away. The trainer class gives you complete control over the hyperparameters and data monitoring.

```py
'''
Quick Start
'''
from ML.proTrain import trainer

gan = trainer(path = 'path/to/data', batch_size = 128, split = [90,10])
gan.train(1) # Trains one epoch
```
Read [ahead](https://github.com/saxenabhishek/ProGan/tree/aks-readme#-inside-the-box) to learn about all the features

## 📺 Preview

<div align="center">
  <img alt="Screenshot" src="docs/preview.png" />
</div>

## 📌 Prerequisites

### 💻 System requirement :

1. You probably need a GPU to train this model
2. Any OS with the correct python version would work
### 💿 Software requirement :

1. Python 3.8 or higher
3. Any text editor of your choice.

## 🔧 Installation 

Step One ( Clone this repo )

Step Two ( Install requirments prefrably in a fresh enviorement )
```
> pip install -r requirements.txt
```

Step Three (Run the demo file)
```
> python main.py
```
## 📦 Inside the box
The 'progan' Package exposes three modules trainer, eva, and definitions.

### Trainer
```py
gan = progan.trainer(
        path: str,
        batch_size: int,
        split: Tuple[int, int, Optional[int]],
        save_dir: str = "ModelWeights/",
        image_dir: str = "ModelWeights/img",
        maxLayerDepth: int = 4,
        merge_samples_const: int = 1,
        lr: Tuple[float, float] = (0.0001, 0.0001),
        loadModel: bool = False,
        plotInNotebook: bool = False,
    ) -> None:
```
#### Parameters
  * **path** - Path to the training folder 
  * **batch_size** - Path to the training folder
  * **split** - A list with ratio split of training and testing data, e.g. [90,10] means Use 90% data for training and 10% for evaluation
  * **save_dir** - Path to the model param and weight save directory
  * **image_dir** - Path to the image save directory
  * **maxLayerDepth** - Max depth of the models after full growth.
  * **merge_samples_Const** - Number of epochs till merging completes after adding a layer.
  * **LR** - tuple with learning rates of the generator and discriminator
  * **loadModel** - If you want to continue training
  * **plotInNotebook** - If set to true, all images are shown instead of being saved 

#### train()
```py
train(epochs: int, display_step: int = 100)
```
* **epochs** - epochs to run for
* **display_step** - Prints stats and runs evaluation of every n number of samples 

#### Step_up()

The original paper has the concept of growing models, which means we add a layer after training the previous layer. Because of how sensitive gans are, we can't add new layers just like that. A constant alpha merges new layers with the model. This constraint makes handling data and models complicated. **Luckily** you don't need to worry about all that. You can call the `step_up` function to grow your model. we take care of everything internally.

```py
 gan.step_up() # if the genarator was generating (4,4) images now it will genarate (8,8) images
```
> Note: Conversly a step_dn funtion is also availblle which does the opposite

#### Plot_trainer()

The dictionary `losses` contains all collected values like Discriminator opinions, other loss terms, etc. Plot trainer function is a simple utilty, it gives you a graph of loss and discriminator outputs. It helps you estimate how the training is going. If the `ShotInNotebook` option is False the graph is saved with a epoch details instead  However, You always have the option to access the values yourself and do your analysis. You can even add to the existing terms and collect more data.

```
gan.losses['gen'] # loss of the genarator
```

#### Saving and loading

These models take a lot of time to train. You'll probably want to run it a few epochs at a time. The trainer class makes this convenient. `params_weig.tar` contains all config and weights to restart training. By default, saves happen after _every epoch_, but you can change that. Besides the Models and optimizer values, the other details which are stored are :
  * **A test noise sample** - so the same test images are genarated everytime
  * **losses** - the dict with all the loss values
  * **LayerDepth** - How many layers are being trained right now
  * **AlphaSpeed** - rate at which alpha updates
  * **Alpha** - Values of current alpha
  * **Epochs** - Epochs completed till now

```py
trainer( 
    ...
    savedir: str = "ModelWeights/", # Save path
    loadmodel: bool = False, # pass this as true to load weights
)
```
#### Loss Funtions
The default loss function is **WGANGP** you can check out the immplementation in `progan\Definitions\loss.py`. We also have the option of using **LSGAN** loss like it is in the paper.

### Eva
```py
eva(path:str ="path/to/model", numrows:int=3, step:int=4)
```
This modules contains function to do generate outputs from saved weights. things like walking in latent space and genarating random images.

#### Parameters
  * **path** - path to the generator function
  * **numrow** - number of rows in grid
  * **steps** - number of steps to take in latent space between two point. 

## 📜 License

ProGAn is available under the MIT license. See the LICENSE file for more info.

## 🤝 Contributing

Please read [`Contributing.md`](https://github.com/saxenabhishek/ProGan/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## ⚙️ Maintainers

| <p align="center">![Abhishek Saxena](https://github.com/saxenabhishek.png?size=128)<br>[Abhishek Saxena]</p> |
| ------------------------------------------------------------------------------------------------------------ |

## 💥 Contributors

<a href="https://github.com/saxenabhishek/ProGan/graphs/contributors">
<img src="https://contrib.rocks/image?repo=saxenabhishek/ProGan" alt="Contributors">
</a>
                                                                                  
## 🚨 Forking this repo
We value keeping this Package open source, but as you all know, _**plagiarism is wrong**_. We spent a non-negligible amount of effort developing and perfected this iteration of the Package, and we are proud of it! All we ask is to not claim this effort as your own.

Feel free to fork this repo. If you do, please give proper credit by linking back to this repo. Refer to this handy [quora post](https://www.quora.com/Is-it-bad-to-copy-other-peoples-code) if you're not sure what to do. Thanks!
