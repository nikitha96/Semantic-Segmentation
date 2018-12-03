# Semantic-Segmentation
## Publications


**"Efficient Semantic Segmentation using Gradual Grouping"**, Nikitha Vallurupalli, Sriharsha Annamaneni, Girish Varma, C V Jawahar, Manu Mathew, Soyeb Nagori, IEEE Embedded Vision Workshop, June 2018, Salt Lake City, UT. [[pdf]](https://arxiv.org/pdf/1806.08522.pdf)
   
## Requirements:

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
* [**Python 3.6**](https://www.python.org/)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 9.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom 
In Anaconda you can install with:
```
conda install numpy matplotlib torchvision Pillow
conda install -c conda-forge visdom
```

If you use Pip (make sure to have it configured for Python3.6) you can install with: 

```
pip install numpy matplotlib torchvision Pillow visdom
```
Each folder contains code for proposed models that are described in the paper.
For instructions please refer to the README on each folder
