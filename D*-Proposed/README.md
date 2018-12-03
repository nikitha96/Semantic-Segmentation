## MODEL: D*-Propsed

PyTorch code for training D*-Proposed model. The code is based initially on [ERFNet](https://github.com/Eromera/erfnet_pytorch) as mentioned in the paper.

## Usage

```
python3 ~/.main.py  --model imgnet_ups --savedir save_imgnet_ups  --dirpath  "path to save dir"  --datadir "path to datadir"  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/model_best.pth.tar
```
Number of epochs and batch size can be set according to your choice.

## Options
For all options and defaults please see the bottom of the "main.py" file. Please set the paths in main.py accordingly.

model_imagenet.py is to extract the pretrained encoder obtained by training the D*-model on Imagenet. "model_best_pth.tar" is the pretrained encoder model.

imgnet_ups.py has the encoder-decoder architecture for D*-Proposed model.

All the results obtained for this model are saved in save_imgnet_ups directory.

## Multi-GPU
To use multiple GPUs, use the CUDA_VISIBLE_DEVICES command:
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py ...
CUDA_VISIBLE_DEVICES=2,3 python3 main.py ...
```
