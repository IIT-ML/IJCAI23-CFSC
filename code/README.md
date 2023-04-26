# CFSC
This repository contains the code used in the paper:

**Context-Aware Feature Selection and Classification**


## Main Requirements

* python 3.9
* numpy 1.23.5
* scikit-learn 1.1.3
* tensorflow 2.8.0
* wittgenstein 0.3.2
* scipy 1.9.3
* matplotlib 3.6.2
* pandas 1.5.2



## Dataset Description
Our dataset directory contains following eight datasets. The first five are real world datasets and the last three are synthetic datasets.

All datasets are preprocessed into csv file with common format.
* Credit
* Company
* Mobile
* NHIS
* Ride
* Synthetic1
* Synthetic2
* Synthetic3

Before running experiments, please put "dataset" folder in the same directory with "code" folder.

## Running Examples:


### **FF**

```
python main.py --model_name=FF --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=Clf_all --num_fold=5 --hidden_dim=16 --hidden_act=tanh --num_epoch=100 --batch_size=32 --learn_rate=1e-2
```

### **LR-L1**

```
python main.py --model_name=LR_l1 --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --lr_C=1
```

### **DT**
```
python main.py --model_name=DT --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --max_depth=-1 --min_samples_split=2 --min_samples_leaf=1 --max_leaf_nodes=-1 --min_imp_dec=0.0
```

### **RL-P / RL-N**
```
python main.py --model_name=<Rule_pos or Rule_neg> --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --max_rules=-1 --max_rule_conds=2 --max_total_conds=-1 --k=2 --n_discretize_bins=10
```

### **BiLSTM**
```
python main.py --model_name=BiLSTM-ATT --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --input_dim=64 --input_act=tanh --lstm_dim=32 --lstm_att_act=sparsemax --att_dim=256 --lambda_att=0.0 --num_epoch=100 --batch_size=32 --learn_rate=1e-2
```

### **BiLSTM-w/FL**
```
python main.py --model_name=BiLSTM-ATT --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --input_dim=64 --input_act=tanh --lstm_dim=32 --lstm_att_act=sparsemax --att_dim=256 --lambda_att=<larger than 0.0> --num_epoch=100 --batch_size=32 --learn_rate=1e-2
```

### **RNP**
```
python main.py --model_name=RNP --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --input_dim=64 --input_act=tanh --att_dim=256 --att_act=tanh --hidden_dim=16 --hidden_act=tanh --lambda_omega=1e-4 --num_epoch=100 --batch_size=32 --learn_rate=1e-2
```

### **LR_Pipeline**
```
python main.py --model_name=LR_FS-CLF --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --fs_C=1 --clf_C=1
```

### **CFSC**
```
python main.py --model_name=CFSC_ft --dataset_name=<dataset name> --type_fl=<feature labeling strategy> --type_learn=FS_MultiL_all --num_fold=5 --input_dim=64 --input_act=tanh --att_dim=256 --att_act=tanh --hidden_dim=16 --hidden_act=tanh --lambda_a=0.5 --num_epoch=100 --batch_size=32 --learn_rate=1e-2
```

