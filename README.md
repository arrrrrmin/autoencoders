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

## Requirements

This repo is only tested for `>= python3.9, <= python3.11`. Please make sure your machine
is running a compatible python version (recommending `3.9`).  

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
git clone https://github.com/arrrrrmin/autoencoders.git
cd autoencoder
poetry install .
poetry run which python
# The last command will show you the environments python path.
# Ok your good to go.
```

### Alternative installation

Clone the repo and cd into it and run:
```
git clone https://github.com/arrrrrmin/autoencoders.git
cd autoencoder
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```
In this case poetry does not play any role, as long as the environment is activated, your fine.

## Usage

#### Run an experiment

Following commands assume you'r in the root of the repo.

* `python3 -m autoencoder.scripts.run` - Open the file and uncomment/edit line and hyperparameters.
* `poetry run python3 -m autoencoder.scripts.run` - Will do the same with poetry running on the machine.
* While the experiment is running execute `tensorboard --logdir logs`, here you have a dashboard to look after your experiment.

You can also do this in `jupyter`, but beaware if something fails the kernel needs a restart.

### Jupyter notebooks

In case the poetry installation is successfull, one can simply use the notebooks locally 
with ``poetry run jupyter notebook``.

When you'r working with a standard python environment you can execute scripts like so:
`jupyter notebook`