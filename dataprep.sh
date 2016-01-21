#!/bin/bash
cd data/fundus

## Resize down to 512 
find . -name "*.jpeg" | xargs -I {} convert {} -resize 512x512 -quality 100 {}

mv train train_512

# resize down to 256 as well
cp -r train_512 train_256
find train_256 -name "*.jpeg" | xargs -I {} convert {} -resize 256x256 -quality 100 {}

