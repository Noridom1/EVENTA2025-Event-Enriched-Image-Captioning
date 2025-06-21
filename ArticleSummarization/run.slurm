#!/bin/bash
#SBATCH --job-name=art_sum_job         # Job name
#SBATCH --output=art_sum_output.log    # Output log
#SBATCH --error=art_sum_error.log      # Error log
#SBATCH --time=24:00:00                # Max wall time
#SBATCH --gres=gpu:1                   # Request 1 GPU (A100 if available)
#SBATCH --mem=36G                      # Memory
#SBATCH --cpus-per-task=4              # CPU cores
#SBATCH --partition=gpu                # Partition name (adjust if needed)

#SBATCH --mail-type=ALL                # Send email on BEGIN, END, FAIL (use NONE, BEGIN, END, FAIL, REQUEUE, etc.)
#SBATCH --mail-user=thinhphuc2005@example.com   # Replace with your actual email

# Load conda environment from your installed Miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate art_sum

# Optional: go to your project directory
cd ArticleSummarization

# Run your Python script
python main.py
