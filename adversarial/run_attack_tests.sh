#!/usr/bin/env bash

for arch in 4 11 12 13 14 15 37 30 21 40 17
do
    python torch-attacks.py --arch nas_init_$arch --model_path ../weights/nas-init/train-NAS-Init-$arch/weights.pt
done

for arch in 1 2 13 24 29 33 34 35 36
do
    python torch-attacks.py --arch nas$arch --model_path ../weights/nas/train-NAS-$arch/weights.pt
done

for arch in 1 2 3 4 6 7 9 11 12 15 16 18 21 26 30 31 32 33 34 36 39 40
do
    python torch-attacks.py --arch ras1_$arch --model_path ../weights/ras1/train-RAS1-$arch/weights.pt
done

for arch in 5 7 10 11 13 14 22 23 24 26 27 28 33 35 36 39 40
do
    python torch-attacks.py --arch ras2_$arch --model_path ../weights/ras2/train-RAS2-$arch/weights.pt
done
