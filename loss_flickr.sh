#!/bin/bash
#!/bin/bash
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mail-user=shashwat.s@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --exclude=gnode063


eval "$(conda shell.bash hook)"
conda activate prior_model_env

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
mkdir -p /scratch/shashwat.s/embed_cache
rm /scratch/shashwat.s/embed_cache/*
cd /scratch/shashwat.s
# wget https://huggingface.co/nousr/conditioned-prior/resolve/main/vit-l-14/laion2b/ema855M.pth?download=true
# mv 'ema855M.pth?download=true.1' diff_prior_ema855M.pth

scp shashwat.s@ada:/share1/shashwat.s/diff_prior_ema855M.pth ./
cd ~/vision_image_compos

scp shashwat.s@ada:/share1/shashwat.s/flickr-8k/* /scratch/shashwat.s/

python loss_based_flickr_eval.py --model_path /scratch/shashwat.s/diff_prior_ema855M.pth --device cuda --num_samples 10 --split 0.01


