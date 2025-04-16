---
layout: default
title: Isaac Lab Tutorial
nav_order: 2
---

# Isaac Lab Tutorial

Welcome to the Isaac Lab Tutorial!

This guide is designed to help you get started with Isaac Lab quickly and easily.  
If you're new to Isaac Sim, this tutorial walks you through the essential installation steps and provides a focused, beginner-friendly experience.

We’ve centered the tutorial around a **quadcopter simulation example**, making it intuitive to learn how to simulate and control flying robots using Isaac Lab.

Our goal is to keep things simple and practical — so you can spend less time for installing and running

---
## Install
To use Isaac Lab, you must first install Omniverse Isaac Sim.

Isaac Lab runs on top of NVIDIA’s simulation engine, Isaac Sim, and relies on it for core functionalities such as robot physics simulation, sensor rendering, and reinforcement learning environment execution. 

Therefore, Isaac Sim must be installed beforehand in order to run Isaac Lab or train any robots.

### Isaac Sim install

First, we have to set up a Python environment for the Isaac Lab project.
We’ll use a conda virtual environment based on Python 3.10 to create an isolated development setup.

'''bash
conda create -b isaaclab python=3.10
conda activate isaaclab

Next, install PyTorch 2.5.1 with CUDA support, depending on your system's CUDA version.

if CUDA 11, install

'''bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

if CUDA 12, install 

'''bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

To install Isaac Sim packages with pip, first we have to update pip

'''bash
pip install --upgrade pip

Then, install Isaac Sim

'''bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

### Isaac Lab install


You can install IsaacLab using pip, either directly from the GitHub repository or by cloning and installing it locally in editable mode.

There are two simple methods to do this:

1. Installing via pip from GitHub  
2. Cloning the repository and installing locally

## Prerequisites

Make sure Isaac Sim is installed and that the Python environment is set up:

```bash
$ source ~/.local/share/ov/pkg/isaac_sim-*/setup_python_env.sh
