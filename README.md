# Introduction to unsupervised learning using autoencoders
An ongoing repository for hands-on lectures for unsupervised learning using auto encoders.

Disclaimer, this repository is heavily influenced by 
[tilman151/ae_bakeoff](https://github.com/tilman151/ae_bakeoff/blob/master/src/run.py) and
[lilianweng's blog](https://lilianweng.github.io/posts/2018-08-12-vae/), which are very good
learning resources. This repo takes the basic autoencoders and tries to visualize some 
detail like e.g. sparse latent code visualizations during training.

## Repo goal

The notebook will show the basic concept and some adaptations of autoencoders and unsupervised
learning. Some definitions are included as well. For inpeth views please see the following 
papers. 

* [1] Geoffrey E. Hinton, and Ruslan R. Salakhutdinov. [“Reducing the dimensionality of data with neural networks."](https://www.science.org/doi/10.1126/science.1127647) Science 313.5786 (2006): 504-507.
* [2] Pascal Vincent, et al. [“Extracting and composing robust features with denoising autoencoders."](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) ICML, 2008.
* [3] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov. [“Improving neural networks by preventing co-adaptation of feature detectors.”](https://arxiv.org/abs/1207.0580) arXiv preprint arXiv:1207.0580 (2012).
* [4] Andrew Ng. [Sparse Autoencoder.](https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf) Lecture notes. 

In case I got something wrong, please open an issue, so I can fix it. This repository is 
work in progress, which means there will be more autoencoders added as soon as I find time to 
learn about them (e.g. VEA / BVEA).

## Installation

### Get poetry

For a detailed guide see the documentation on poetry. 

```
# Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -
# Windows (Powershell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Install this repo

Invoke poetry install, the rest will be taken care of.

```
git clone autoencoder
cd autoencoder
poetry install .
poetry run which python
# The last command will show you the environments python path.
# Ok your good to go.
```

## Usage

### Jupyter notebooks

In case the poetry installation is successfull, one can simply use the notebooks locally 
with ``poetry run jupyter``.

### Local development

In case you fork the repo, there's a `scripts` folder in the root directory. 
Call for example ``poetry run python -m autoencoders.scripts.download_mnist``.
Experiments can be reproduced using the main entry in 
[``autoencoders/scripts/run``](autoencoders/scripts/run.py).
