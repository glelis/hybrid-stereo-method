#!/bin/bash

#to compile: chmod +x copy_images.sh
#run: ./copy_images.sh 

# Source directory containing images
DIR="/home/lelis/Documents/Projetos/hybrid-stereo-method/data/raw/hybrid_stereo/2025-02-09-stF-melon24-amb0-glo0/"


mkdir -p $DIR/images_mf;

for l in 000 001 002 003 004 005 006 007 008 009 010 011; do
	mkdir -p $DIR/images_mf/L_$l;
	mkdir -p $DIR/images_mf/L_$l/sharp;
	mkdir -p $DIR/images_mf/L_$l/images;
	cp -rv "$DIR/stF-melon24/0512x0384-hs01-kr10/L$l-amb0.00-glo0.00/sharp/" "$DIR/images_mf/L_$l/sharp/";
		
	for f in 026 033 041 048 056 063 071 078 086 093; do
		cp -v "$DIR/stF-melon24/0512x0384-hs01-kr10/L$l-amb0.00-glo0.00/zf$f.2500-df015.0000/sVal.png" "$DIR/images_mf/L_$l/images/sVal_zf$f.png";
		cp -v "$DIR/stF-melon24/0512x0384-hs01-kr10/L$l-amb0.00-glo0.00/zf$f.7500-df015.0000/sVal.png" "$DIR/images_mf/L_$l/images/sVal_zf$f.png";
	done
done

