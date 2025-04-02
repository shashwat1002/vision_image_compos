

mkdir -p /scratch/shashwat.s/.cache

python download_coco_fr.py

cd /scratch/shashwat.s/

zip coco_cache.zip .cache

scp coco_cache.zip shashwat.s@ada.iiit.ac.in:/share1/shashwat.s/

