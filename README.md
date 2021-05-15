# ✨ ProGAN

<div align="center">
  <img alt="GitHub Community SRM Logo" src="assets\progan-ep41.gif" height="" />
</div>
<div align="center">
  
  
##### Training 40 epochs on the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html) dataset
  
###### [top-bottom] 3 numbers on top indicates `Epochs, Depth, and Alpha-value`, respectively. The first nine image blocks are fake-generated data; the rest are actual samples.

</div>

<br>

## 💡 Project Description

This project is a reimplementation of the paper[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)  by Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen. Check out this [article](https://towardsdatascience.com/progan-how-nvidia-generated-images-of-unprecedented-quality-51c98ec2cbd2) for a recap on how GANs work and how progressive gans are different.

This repository is developed like a python package, so it is easy to quickly start training on any dataset. Everything you need is an import away. The trainer class gives you complete control over the hyperparameters and data monitoring.

```py
'''
Quick Start
'''
from ML.proTrain import trainer

gan = trainer(path = 'path/to/data', batch_size = 128, split = [90,10])
gan.train(1) # Trains one epoch
```
These models take a lot of time to train. You'll probably want to run it a few epochs at a time. The trainer class makes this convenient. `params_weig.tar` contains all config and weights to restart training. By default, saves happen every epoch, but you can change that.

```py

'''
default values
'''
trainer( 
    ...
    savedir: str = "ModelWeights/", # Save path
    loadmodel: bool = False, # pass this as true to load weights
)
```


## 📺 Preview

<div align="center">
  <img alt="Screenshot" src="docs/preview.png" />
</div>

## 📌 Prerequisites

### 💻 System requirement :

1. Any system with basic configuration.
2. Operating System : Any (Windows / Linux / Mac).

### 💿 Software requirement :

1. Updated browser
2. Node.js installed (If not download it [here](https://nodejs.org/en/download/)).
3. Any text editor of your choice.

## Installation 🔧

Step One (explain the step as well!)

```
$ script one
```

Step Two (explain the step as well!)

```
$ script two and you get it
```

## 📦 Inside the box

Add your `Wiki` here. Remove this section if not there.

## 📜 License

`{Project name}` is available under the MIT license. See the LICENSE file for more info.

## 🤝 Contributing

Please read [`Contributing.md`](https://github.com/SRM-IST-KTR/template/blob/main/Contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## ⚙️ Maintainers

| <p align="center">![Abhishek Saxena](https://github.com/saxenabhishek.png?size=128)<br>[Abhishek Saxena]</p> |
| ------------------------------------------------------------------------------------------------------------ |

## 💥 Contributors

  <!-- replace 'githubsrm' with your repository name -->
<a href="https://github.com/saxenabhishek/ProGan/graphs/contributors">
<img src="https://contrib.rocks/image?repo=saxenabhishek/ProGan" alt="Contributors">
</a>
                                                                                  
## 🚨 Forking this repo

Many people have contacted us asking if they can use this code for their own websites. The answer to that question is usually "yes", with attribution. There are some cases, such as using this code for a business or something that is greater than a personal project, that we may be less comfortable saying yes to. If in doubt, please don't hesitate to ask us.

We value keeping this site open source, but as you all know, _**plagiarism is bad**_. We spent a non-negligible amount of effort developing, designing, and trying to perfect this iteration of our website, and we are proud of it! All we ask is to not claim this effort as your own.

So, feel free to fork this repo. If you do, please just give us proper credit by linking back to our website, https://githubsrm.tech. Refer to this handy [quora post](https://www.quora.com/Is-it-bad-to-copy-other-peoples-code) if you're not sure what to do. Thanks!
