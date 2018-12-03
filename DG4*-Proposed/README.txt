MODEL: DG4*-Propsed

Please refer "https://docs.google.com/spreadsheets/d/1dW-s03Z_kAOt0eey5drzSAEwVu7y_Ag7Rm338FuFVjU/edit#gid=192516595" to understand the model better and for FLOP calculations

This folder contains code for DG4*-Proposed model.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DATASET:

Please create datadir containing cityscapes. Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the cityscapes scripts and use the conversor to generate trainIds from labelIds. cityscapes folder should contain the complete dataset.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HOW TO RUN:

python3 ~/DG_main.py  --model DGstar_model --savedir DG2star  --dirpath "path to dir"   --datadir "path to dataset"  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/G2_Imgnet_Chkpt/model_best.pth.tar --groups 4

number of epochs and batch size can be set according to your choice.

Please refer script_G4.sh. Please set the paths in DG_main.py accordingly.

DGstar_model_imagenet.py is to extract the pretrained encoder obtained by training the DG2*-model using gradual grouping on Imagenet. "model_best_pth.tar" is the pretrained encoder model obtained after training on Imagenet in G4_Imgnet_Chkpt folder.

DGstar_model.py has the encoder-decoder architecture for DG4*-Proposed model.

All the results obtained for this model are saved in DG4star directory.
