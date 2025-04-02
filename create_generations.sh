#!/bin/bash
#!/bin/bash
#SBATCH --mem-per-cpu=3076
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mail-user=shashwat.s@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --exclude=gnode063
#SBATCH --nodelist=gnode042

eval "$(conda shell.bash hook)"
conda activate image_stuff

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
mkdir -p /scratch/shashwat.s/images

cd notebooks
python dalle_trial.py --dataset_frac  0.04

scp -r /scratch/shashwat.s/images shashwat.s@ada:/share1/shashwat.s/gen_images

