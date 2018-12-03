

## MODEL: DG4*-Propsed

PyTorch code for training DG4*-Proposed model.

## Example commands

```
python3 ~/DG_main.py  --model DGstar_model --savedir DG2star  --dirpath "path to dir"   --datadir "path to dataset"  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/G2_Imgnet_Chkpt/model_best.pth.tar --groups 4

```
Number of epochs and batch size can be set according to your choice.

## Options
For all options and defaults please see the bottom of the "DG_main.py" file. Please set the paths in main.py accordingly.

DGstar_model_imagenet.py is to extract the pretrained encoder obtained by training the DG4*-model using gradual grouping on Imagenet. "model_best_pth.tar" is the pretrained encoder model obtained after training on Imagenet in G4_Imgnet_Chkpt folder.

DGstar_model.py has the encoder-decoder architecture for DG4*-Proposed model.

All the results obtained for this model are saved in DG4star directory.


## Multi-GPU
To use multiple GPUs, use the CUDA_VISIBLE_DEVICES command:
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py ...
CUDA_VISIBLE_DEVICES=2,3 python3 main.py ...
```
