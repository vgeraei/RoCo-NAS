#!/usr/bin/env bash

for arch in 21
do
    python torch-attacks.py --arch ras1_21 --model_path ../weights/adv-train/train-ras1_21_adv/weights.pt
done


