# Graph Neural Network for Atom Mapping in Chemical Reaction

This repository is an author's implementation of the paper Graph Neural Network for Atom Mapping in Chemical Reaction. 


## Overview

The authors propose a method that is capable of predicting outcomes and finding atom mapping at the same time. 
Two specific tasks are solved in parallel. Atoms of the main product and centers of the reaction are found. Centers of
the reaction are atoms of the main product, which change the configuration in the reaction. The configuration of an
atom is a superposition of characteristics of the atom and adjacent bonds.  In terms of graph theory, both tasks are
node-classification in a disconnected graph of source molecules. The novel neural network MolsNet solves
node-classification tasks. Atoms of the main product and centers of the reaction determine the outcome in the majority 
of reactions because they have less than three centers. 
![](imgs/problem_statement.png)

The method structure consists of several blocks. Firstly, each atom is mapped to a
 real vector according to its characteristics in the molecule. The model is capable of using any known numerical 
 characteristics of atoms. Secondly, the vectors are updated with Relational Graph Convolution Neural Network (RGCNN).
 The authors offer to use extended molecular graphs with molecule's and reaction's level nodes to enable passing 
 information across different molecules. The RGCNN generalizes Graph Convolution Neural Network for graphs with 
 different edge types that correspond to chemical bonds. Then, the Transformer encoder processes the vectors. The block
 simulates intermolecular interaction, which is a mechanism of chemical reactions. Finally, the Fully-connected neural
 network (FCNN) gives probabilities for each atom in the node classification problems.
![](imgs/architecture.png)

Compared with other recent studies, MolsNet has several novel aspects in terms of architecture of neural networks. 
MolsNet generalizes the graph convolution neural network for the disconnected graph of molecules. The natural structure
 of the MolsNet architecture is suitable to add information about molecules and atoms: characteristics of atoms, 
 types of chemical bonds.  
 
The authors investigate vector representations of reactions. The best model demonstrates that pseudo-nodes in the 
extended graph of source molecules learn chemical information about the whole reaction. Similar representation 
correspond chemical reactions which have a similar mechanism.  
 ![](imgs/nearest_rections.png)
 
 TSNE MAP shows that space of reaction's representation contains information about the class of reaction.
 
 ![](imgs/tsne_minor.png)
 
 ## Requiremnts
 
 Linux machine with Nvidia-1080Ti, Python 3.7.7 was used for the experiments with the folowing packages:
 
 ```
dgl-cu101==0.4
Django==3.0.5
matplotlib==3.2.1
numpy==1.18.2
pandas==1.0.3
PyYAML==5.3.1
scikit-learn
scipy
seaborn==0.10.0
torch==1.4.0
tqdm==4.45.0
rdkit
```

 ## Data preprocessing
 Our custom data preprocessor takes pandas dataframes stored in pickle. The dataframes should consist column 'smarts' 
 with canonical reactions, and different other columns which will be stored in the processed dataset. 
 
 To create dataset, you should use the following commands:
 
 ```
cd scripts
python -u process_dataset.py --raw_dataset 'path to raw pickles directory' --save_path 'save directory'
```
 
 ## Training
 
 To train a model, you should use the following commands:
  ```
cd scripts
python -u experiment.py --config path to YAML configuration --device name of using device
```

Examples of ```.yml``` configurations are in ```experiments``` directory.
 
 ## Online demo
 
We created an online demo which source code is in ```server``` directory. The demo is available at http://93.175.29.159:8000/.