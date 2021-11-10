#!/usr/bin/env bash

for arch in 22 5 19 35 28 34 8 25 38 33 37 26
do
    python test_dqa.py --arch baseline_nas_$arch --model_path ../weights/baseline-nas/train-Baseline-NAS-$arch/weights.pt
done
