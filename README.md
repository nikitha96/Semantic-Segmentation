# Semantic-Segmentation
Requirements:

    The Cityscapes dataset: Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the cityscapes scripts and use the conversor to generate trainIds from labelIds
    
    Python 3.6
    PyTorch: Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 9.0).
    Additional Python packages: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)

In Anaconda you can install with:

conda install numpy matplotlib torchvision Pillow
conda install -c conda-forge visdom

If you use Pip (make sure to have it configured for Python3.6) you can install with:

pip install numpy matplotlib torchvision Pillow visdom

This folder contains codes for 3 models D*-Proposed, DG2*-Proposed and DG4*-Proposed as described in the paper.
