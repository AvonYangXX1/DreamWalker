#!/usr/bin/bash
#SBATCH -c 1
#SBATCH -t 5-00:05 
#SBATCH -p medium
#SBATCH --mem=300G
#SBATCH -o siy105_%j.out
#SBATCH -e siy105_%j.err

module load python/3.8
module load conda2/4.2.13

python unet_diffusion_ver1.py
 
