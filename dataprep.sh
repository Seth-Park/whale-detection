#!/bin/bash
cd data/whale

## Resize down to 512 
find . -name "*.jpg" | xargs -I {} convert {} -resize 512x512 -quality 100 {}

mv imgs train_512

# resize down to 256 as well
cp -r train_512 train_256
find train_256 -name "*.jpg" | xargs -I {} convert {} -resize 256x256 -quality 100 {}

