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
#SBATCH --output=042423_CNN_simple.out


echo "Starting"
date

time python CNN.py


echo "Finished"
date

exit;
