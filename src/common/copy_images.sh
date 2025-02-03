#!/bin/bash

#to compile: chmod +x copy_images.sh
#run: ./copy_images.sh 

# Source directory containing images
DIR="/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/photometric_stereo/ex23/"


mkdir $DIR/images;

for i in 00 01 02 03 04 05 06 07 08 09 10 11;
	do cp -v "$DIR/2025-02-03-stB-noise01/stB-noise01/0512x0384-hs01-kr10/L0$i-amb0.0000/sharp/sVal.png" "$DIR/images/image$i.png"; 
done

echo "Copied images!"

