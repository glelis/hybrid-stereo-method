#!/bin/bash

#to compile: chmod +x copy_images.sh
#run: ./copy_images.sh 

# Source directory containing images
DIR="/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/2025-03-08-stQ-melon24-amb0.00-glo0.50/"
'stQ-melon14-amb0.00-glo0.50/0512x0384-hs01-kr10/L000/sharp'


mkdir -p $DIR/images_ps;
src/common/copy_images.sh
for i in 00 01 02 03 04 05 06 07 08 09 10 11; do 
	
	cp -v "$DIR/stQ-melon14-amb0.00-glo0.50/0512x0384-hs01-kr10/L0$i/sharp/sVal.png" "$DIR/images_ps/image$i.png"; 
done

echo "Copied images to images_ps directory!"



