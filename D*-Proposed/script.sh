#!/bin/bash
#SBATCH -A research
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --qos=medium            # medium
#SBATCH --partition=long              # Mention partition-name. default
#SBATCH --gres=gpu:2      # N number of GPU devices.
#SBATCH --mem=64G
#SBATCH --nodelist=gnode28
#SBATCH --mem-per-cpu=4000


module add cuda/9.0
module add cudnn/7-cuda-9.0

export CUDA_VISIBLE_DEVICES=2,3

source ~/.virtualenvs/Experiments/bin/activate

python3 ~/Experiments_pytorch/Project_mine/main.py  --model imgnet_ups --savedir save_imgnet_ups  --dirpath  ~/Experiments_pytorch/Project_mine/ --datadir /ssd_scratch/cvit/cityscapes/  --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder ~/Experiments_pytorch/Project_mine/model_best.pth.tar
