#!/bin/bash
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=LP_n18_s2
#SBATCH --output=/home/frwe188h/DistributedFunSearch/analysis/baselines/LP_n9_s1_ids_%j.out
#SBATCH --error=/home/frwe188h/DistributedFunSearch/analysis/baselines/LP_n9_s1_ids_%j.err

# LP upper bound — no graph files needed, built from combinatorics
# Uses scipy HiGHS solver (no license needed)

module purge
module load release/25.06
module load GCC/13.3.0
module load Python
module load SciPy-bundle/2024.05

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "===== Job Info ====="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Memory:       ${SLURM_MEM_PER_NODE} MB"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "===================="

cd /home/frwe188h/DistributedFunSearch/analysis/baselines

srun python upper_bound_LP.py \
    --n 9 \
    --s 1 \
    --q 4 \
    --edit \
    --solver scipy \
    --cache-dir /data/horse/ws/frwe188h-disfun/Graphs \
    --verbose \
    --json 2>&1 | tee LP_n9_s1_ids_result.log

echo "End time: $(date)"
