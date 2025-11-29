#!/bin/sh
#SBATCH --job-name=mswe_gnn_train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=4320

. venv/bin/activate

echo "========== New HEC-RAS Data =========="
srun python main.py --config 'configs/hecras_config.yaml'
