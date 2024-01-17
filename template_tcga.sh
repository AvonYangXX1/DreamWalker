#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 8:00:00                         # Runtime in D-HH:MM format
#SBATCH -p short                            # Partition to run in
#SBATCH --mem=1G                          # Memory total in MiB (for all cores)
#SBATCH -o /n/data2/hms/dbmi/kyu/lab/shl968/GBM_PCNSL_classification/image_patching_script/slurm_out/log_%j_[imgfile_short].log                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/data2/hms/dbmi/kyu/lab/shl968/GBM_PCNSL_classification/image_patching_script/slurm_out/log_%j_[imgfile_short].log                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like
source activate deeplearning

python WSI_tile_extraction.py
 --infile [imgfile] --proj [proj_name] --params [params]
