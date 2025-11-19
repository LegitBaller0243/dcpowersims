#!/bin/bash
#SBATCH --job-name=DCPowerSims
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# Create or activate virtual environment
module load python/3.10
if [ ! -d "sarimax_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv sarimax_env
    source sarimax_env/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install --extra-index-url=https://pypi.nvidia.com -r requirements.txt
else
    echo "Using existing virtual environment..."
    source sarimax_env/bin/activate
fi

python -u sarimax_script.py
deactivate
