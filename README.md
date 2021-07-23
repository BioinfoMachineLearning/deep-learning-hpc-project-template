### Deep learning project seed

Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals

The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

 
---

<div align="center">    

# Your HPC Project Name

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/amorehead/deep-learning-hpc-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>

## Description

What it does

## How to run

First, create a Conda environment for the project:

```bash
# Clone project   
git clone https://github.com/YourGithubName/deep-learning-hpc-project-template

# Install project   
cd deep-learning-hpc-project-template

# (If on HPC cluster) Load 'open-ce' module
module load open-ce/1.1.3-py38-0

# (If on HPC cluster) Clone Conda environment into this directory using provided 'open-ce' environment:
conda create --name dlhpt --clone open-ce-1.1.3-py38-0

# (If on HPC cluster - Optional) Create Conda environment in a particular directory using provided 'open-ce' environment:
conda create --prefix MY_VENV_DIR --clone open-ce-1.1.3-py38-0

# (Else, if on local machine) Set up Conda environment locally
conda env create --name dlhpt -f environment.yml

# (Else, if on local machine - Optional) Create Conda environment in a particular directory using local 'environment.yml' file:
conda env create --prefix MY-VENV-DIR -f environment.yml

# Activate Conda environment located in the current directory:
conda activate dlhpt

# (Optional) Activate Conda environment located in another directory:
conda activate MY-VENV-DIR

# (Optional) Deactivate the currently-activated Conda environment:
conda deactivate

# (If on local machine - Optional) Perform a full update on the Conda environment described in 'environment.yml':
conda env update -f environment.yml --prune

# (Optional) To remove this long prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
```

(If on HPC cluster) Install all project dependencies:

```bash
# Install project as a pip dependency in the Conda environment currently activated:
pip3 install -e .

# Install external pip dependencies in the Conda environment currently activated:
pip3 install -r requirements.txt

# Install pip dependencies used for unit testing in the Conda environment currently activated:
pip3 install -r tests/requirements.txt
 ```

Then, navigate to any file and run it:

 ```bash
# Module folder
cd project

# Run module (example: mnist as your main contribution)   
python lit_image_classifier.py    
```

## Imports

This project is set up as a package which means you can now easily import any file into any other file like so:

```python
from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from project.lit_mnist import LitClassifier

# Model
model = LitClassifier()

# Data
dataset = MNIST('', train=True, download=False, transform=transforms.ToTensor())
mnist_test = MNIST('', train=False, download=False, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32, num_workers=1)
val_loader = DataLoader(mnist_val, batch_size=32, num_workers=1)
test_loader = DataLoader(mnist_test, batch_size=32, num_workers=1)

# Train
trainer = Trainer()
trainer.fit(model, train_loader, val_loader)

# Test using the best model!
trainer.test(test_dataloaders=test_loader)
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   