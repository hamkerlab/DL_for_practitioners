# Jupyter notebooks for the 'deep learning for practitioners' course

## Dependencies

 - Python (>= 3.11)
 - notebook (>=7.3.2)
 - numpy (>= 2.1.2)
 - matplotlib (>=3.10.1)
 - pandas (>=2.2.3)
 - scikit-learn (>=1.6.1)
 - torch (>= 2.7.0)
 - tqdm (>=4.67.19)
 - ipywidgets (>= 8.1.5)
 - seaborn (>=0.12.2)

In the repo you find also a *yml file that can be used to initialize a conda-environment.


### Install required packages

The best way to install the required packages is to use miniconda. You can get the actual and also older version directly from the [anaconda website](https://www.anaconda.com/docs/getting-started/miniconda/install).

After the installation of miniconda, create a new environment, for example with:

```

conda create -n microcredential

```

Here, _microcridential_ is the name of the example environment

Now activate the environment with:

```
conda activate microcredential
```

Please make sure that the environment is active throughout the installation process of the packages and while working with the notebooks.

we first install Python:

```
conda install python
```

If you want, you can specify the Python version you want to use. For example:

```
conda install python=3.11
```

installs Python version 3.11.


Together with Python, conda also installs pip in the environment, what we use to install the other necessary packages.
As PyTorch has to be installed either in a CPU only version, on in a GPU version (which also depends on your CUDA driver) we will first install all other packages mentioned above. 
This can be done with one line:

```
pip install notebook numpy seaborn pandas scikit-learn tqdm ipywidgets
```

Pip will automatically install all other necessary Python packages.

If all packages have been successfully installed, proceed to install PyTorch.

### Install PyTorch

Check your operating system, decide if you want to run it on CPU only or with GPU support, and for the latter also check your CUDA version.
Then go to the [PyTorch website](https://pytorch.org/get-started/locally/) and select under START LOCALLY the conditions you want to use to install PyTorch.
It will then generate the pip command to install PyTorch and all necessary packages.
**NOTE** Did not use the _pip3_ command, use _pip_ instead.

Here is an example pip line to install it with CUDA 12.8: 

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If you want to use pretrained models from [timm](https://timm.fast.ai/), you can install it with:

```
pip install timm
```

## Notebooks


1. Qickstart to PyTorch
2. Deep neural networks
   
   2.1. Introduction to multilayer perceptron (MLP)

   2.2. Introduction to convolutional neural networks (CNN)

   2.3. CNN with Imagenette

   2.4. ResNet 

   2.5. YOLO 
