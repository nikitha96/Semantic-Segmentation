#!/bin/bash
#SBATCH -A research
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --qos=medium            # medium
#SBATCH --partition=long              # Mention partition-name. default
#SBATCH --gres=gpu:2      # N number of GPU devices.
#SBATCH --mem=64G
#SBATCH --nodelist=gnode15
#SBATCH --mem-per-cpu=4000


module add cuda/9.0
module add cudnn/7-cuda-9.0

export CUDA_VISIBLE_DEVICES=0,1

source ~/.virtualenvs/Experiments/bin/activate

python3 ~/Experiments_pytorch/Project_mine/DG_main.py  --model DGstar_model --savedir DG2star  --dirpath  ~/Experiments_pytorch/Project_mine/	--datadir /ssd_scratch/cvit/cityscapes/  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/G2_Imgnet_Chkpt/model_best.pth.tar --groups 2
