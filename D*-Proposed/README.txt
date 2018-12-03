
MODEL: D*-Propsed

Please refer "https://docs.google.com/spreadsheets/d/1dW-s03Z_kAOt0eey5drzSAEwVu7y_Ag7Rm338FuFVjU/edit#gid=192516595" to understand the model better and for FLOP calculations

This folder contains code for D*-Proposed model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DATASET:

Please create datadir containing cityscapes. Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the cityscapes scripts and use the conversor to generate trainIds from labelIds. cityscapes folder should contain the complete dataset.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HOW TO RUN:

python3 ~/.main.py  --model imgnet_ups --savedir save_imgnet_ups  --dirpath  "path to save dir"  --datadir "path to datadir"  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/model_best.pth.tar

number of epochs and batch size can be set according to your choice.

Please Refer script.sh. Please set the paths in main.py accordingly.

model_imagenet.py is to extract the pretrained encoder obtained by training the D*-model on Imagenet. "model_best_pth.tar" is the pretrained encoder model.

imgnet_ups.py has the encoder-decoder architecture for D*-Proposed model.

All the results obtained for this model are saved in save_imgnet_ups directory.
