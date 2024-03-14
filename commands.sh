#!/bin/bash
#

# python train_flow.py --config configs/train_ANN.yml --path_mlflow=ANN --regularizer_weight_voltage=2.0e-6 --regularizer_weight_threshold=2.0e-6  # train ANN 
# python train_flow.py --config configs/train_SNN.yml --path_mlflow=SNN --regularizer_weight_voltage=2.5e-7 --regularizer_weight_threshold=2.5e-7 # train SNN 

python eval_flow.py 9d6a4998c6df42129333b97fd7839783 --config configs/eval_MVSEC_ANN.yml --path_mlflow=trained_networks # test our trained ANN 
python eval_flow.py a372ed0a321e46a2b87acc4358dad0dd --config configs/eval_MVSEC_SNN.yml --path_mlflow=trained_networks # test our trained SNN 