# MODEL: D*-Propsed

PyTorch code for training D*-Proposed model.

## Options
For all options and defaults please see the bottom of the "main.py" file. Required ones are --savedir (name for creating a new folder with all the outputs of the training) and --datadir (path to cityscapes directory).

## Example commands

```
python3 ~/.main.py  --model imgnet_ups --savedir save_imgnet_ups  --dirpath  "path to save dir"  --datadir "path to datadir"  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/model_best.pth.tar
```
Number of epochs and batch size can be set according to your choice.

## Options
For all options and defaults please see the bottom of the "main.py" file. Required ones are --savedir (name for creating a new folder with all the outputs of the training) and --datadir (path to cityscapes directory).
Please Refer script.sh. Please set the paths in main.py accordingly.

model_imagenet.py is to extract the pretrained encoder obtained by training the D*-model on Imagenet. "model_best_pth.tar" is the pretrained encoder model.

imgnet_ups.py has the encoder-decoder architecture for D*-Proposed model.

All the results obtained for this model are saved in save_imgnet_ups directory.

## Multi-GPU
If you wish to specify which GPUs to use, use the CUDA_VISIBLE_DEVICES command:
```
CUDA_VISIBLE_DEVICES=0 python main.py ...
CUDA_VISIBLE_DEVICES=0,1 python main.py ...
```
