#!/usr/bin/env bash

for arch in 5 28 8 19 22 25 26 33 34 35 37 38
do
    python validation/train.py --net_type macro --cutout --batch_size 128 --epochs 120 --arch baseline_nas_$arch --save Baseline-NAS-$arch
    python validation/adv-train.py --net_type macro --cutout --batch_size 128 --epochs 120 --arch ras_21 --save ras1_21_adv
done
