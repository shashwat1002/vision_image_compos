#!/bin/bash
#!/bin/bash
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --mail-user=shashwat.s@research.iiit.ac.in
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate image_stuff

mkdir -p /scratch/shashwat.s/.cache

python download_coco_fr.py

cd /scratch/shashwat.s/

zip -r coco_cache.zip .cache

scp coco_cache.zip shashwat.s@ada.iiit.ac.in:/share1/shashwat.s/

