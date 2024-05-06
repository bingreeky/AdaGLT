# Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.24.4
scipy            1.10.1
torch            1.12.1+cu113
torch-geometric  2.3.1
ogb              1.3.6
dgl-cu101        0.4.3
```

# Usage

## GCN
Please `cd Ada_gcn` first, then use the following commands to check the extreme graph/weight sparsity that **AdaGLT** is capable of achieving. 
```
python -u main.py --dataset cora --num_layers 2 --coef 0.005 --spar_adj --target_adj_spar 27
python -u main.py --dataset cora --num_layers 2 --spar_wei --target_wei_spar 93

python main.py --dataset citeseer --num_layers 2 --spar_adj --e1 9e-5 --total_epoch 600 --pretrain_epoch 200 --coef 0.01 --target_adj_spar 43
python main.py --dataset citeseer --num_layers 2 --spar_wei --target_wei_spar 97 --e2 2e-3 --total_epoch 400 

python -u main.py --dataset pubmed --num_layers 2 --coef 0.005 --spar_adj --target_adj_spar 27
python -u main.py --dataset pubmed --num_layers 2 --spar_wei --target_wei_spar 97
```

For deep GCN scenarios, execute the following commands.
```
python -u main.py --dataset cora --task_type full --use_bn --num_layers 4/8/12/16 --coef 0.005 --e1 2e-3 --spar_adj --target_adj_spar 40

python -u main.py --dataset citeseer --task_type full --use_bn --num_layers 4/8/12/16 --coef 0.005 --e1 2e-3 --spar_adj --target_adj_spar 40

python -u main.py --dataset pubmed --task_type full --use_bn --num_layers 4/8/12/16 --coef 0.005 --e1 2e-3 --spar_adj --target_adj_spar 40
```

## ResGCN
For ResGCN demonstration, please `cd Ada_gcn` first.
```
python -u main.py --dataset cora --task_type full --use_bn --use_res --num_layers 4/8/12/16 --coef 0.005 --spar_adj --target_adj_spar 40

python -u main.py --dataset citeseer --task_type full --use_bn --use_res --num_layers 4/8/12/16 --coef 0.005 --spar_adj --target_adj_spar 40

python -u main.py --dataset pubmed --task_type full --use_bn --use_res --num_layers 4/8/12/16 --coef 0.005--spar_adj --target_adj_spar 40
```

## GIN
For GIN demonstration, please `cd Ada_gin` first. 
```
python -u main.py --dataset cora --num_layers 2 --coef 0.01 --spar_adj --target_adj_spar 22
python -u main.py --dataset cora --num_layers 2 --spar_wei --target_wei_spar 96

python -u main.py --dataset citeseer --num_layers 2 --coef 0.05 --spar_adj --targe_adj_spar 42
python -u main.py --dataset citeseer --num_layers 2 --spar_wei --target_wei_spar 96

python -u main.py --dataset pubmed --num_layers 2 --coef 0.2 --spar_adj --target_adj_spar 50
python -u main.py --dataset pubmed --num_layers 2 --spar_wei --target_wei_spar 96
```

## GAT
For GAT demonstration, please `cd Ada_gat` first.
```
python -u main.py --dataset cora --num_layers 2 --pretrain_epoch 50 --spar_adj --target_adj_spar 72 
python main.py --dataset cora --spar_wei --num_layers 2 --total_epoch 200 --pretrain_epoch 50 --target_wei_spar 98

python -u main.py --dataset citeseer --num_layers 2 --pretrain_epoch 50 --spar_adj --target_adj_spar 82
python -u main.py --dataset citeseer --spar_wei --num_layers 2 --total_epoch 200 --pretrain_epoch 50 --target_wei_spar 93

python -u main.py --dataset pubmed --num_layers 2 --pretrain_epoch 50 --spar_adj --target_adj_spar 82
python -u main.py --dataset pubmed --spar_wei --num_layers 2 --total_epoch 200 --pretrain_epoch 50 --target_wei_spar 98
```

For deep GNN scenerios, execute the following commands.
```
python -u main.py --dataset cora --num_layers 4/8/12/16 --coef 0.005 --spar_adj --target_adj_spar 40

python -u main.py --dataset citeseer --num_layers 4/8/12/16 --coef 0.005 --spar_adj --target_adj_spar 50 

python -u main.py --dataset pubmed --use_bn --num_layers 4/8/12/16 --coef 0.005 --e1 2e-3 --spar_adj --target_adj_spar 50
```

