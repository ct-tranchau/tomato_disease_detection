#! /bin/bash
###
###Script used to run the pipeline
###
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p p100_normal_q
#SBATCH -A introtogds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnchau@vt.edu
#SBATCH --output=042423_VGG16_woTrain_new.out


echo "Starting"
date

time python VGG16_woTrain.py


echo "Finished"
date

exit;
