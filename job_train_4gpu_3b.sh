#!/bin/bash
#PBS -q gpu_as
#PBS -P gs_ccds_boan
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /projects_vol/gp_boan/vla-routing/virl39k_4gpu_3b.log
set -e
module load anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easyr1_virl39k_jh
cd /projects_vol/gp_boan/vla-routing
git pull
bash scripts/train_virl39k_4gpu_3b.sh
