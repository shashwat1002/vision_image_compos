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
#SBATCH --nodelist=gnode090


eval "$(conda shell.bash hook)"
conda activate image_stuff

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

mkdir -p /scratch/shashwat.s/embed_cache

cd /scratch/shashwat.s 

scp shashwat.s@ada.iiit.ac.in:/share1/shashwat.s/coco_cache.zip ./

unzip coco_cache.zip


export TRANSFORMERS_CACHE=/scratch/shashwat.s/.cache
export HF_HOME=/scratch/shashwat.s/.cache

cd ~/vision_image_compos

python mapping_train.py --text_model roberta-base --image_model facebook/dinov2-base --mapping_style attention_mlp --num_heads 8 --num_attn_layers 8 --train_batch 32 --dev_batch 32 --test_batch 32 --device cuda --cache_dir /scratch/shashwat.s/embed_cache --hiddens 512 256 128 256 512 

cd /scratch/shashwat.s

zip -r embed_cache.zip embed_cache

scp embed_cache.zip shashwat.s@ada.iiit.ac.in:/share1/shashwat.s/embed_cache_map.zip 


