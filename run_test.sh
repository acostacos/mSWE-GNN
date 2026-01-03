
#!/bin/sh
#SBATCH --job-name=mswe_gnn_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=720

. venv/bin/activate

echo "========== New HEC-RAS Data =========="
srun python test_model.py --config 'configs/hecras_config.yaml'

echo "========== Original Data =========="
srun python test_model.py --config 'config.yaml'

