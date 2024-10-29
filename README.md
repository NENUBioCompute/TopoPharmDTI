# TopoPharmDTI: Improving Interactions Prediction by Enhanced Deep Learning Representa-tion for Both Drug and Target Molecules
## Model architecture
![image](https://github.com/NENUBioCompute/TopoPharmDTI/blob/main/Figure/Model%20architecture.png)

## Setup and dependencies
dependencies:
+ python 3.9
+ pytorch >= 2.0.1
+ numpy
+ pandas
+ RDkit
+ pyg


##  Source codes:
+ cpi_model.py: TopoPharmDTI model file
+ smiles2graph.py: Generate drug features
+ smiles2subgraph.py: Generate drug subgraph features
+ protein_embedding.py: Generate protein features
+ train.py: train a TopoPharmDTI model.


## Model and Datasets
Trained models and datasets is now available freely at https://drive.google.com/file/d/1SX8xc_jHDnGyPRKm6m6_oe_H9hXHkfyN/view?usp=sharing.

## Run
+ generate drug and protein features：Generate two types of drug features through smiles2Graph.py and smiles2Subgraph.py, respectively
+ Generate protein features by running protein_embedding.py
+ begain train

````
    python train.py
````
+ inference:Load the trained model for testing on the testset
````
    python prediction.py
````
## Case study
![image](https://github.com/NENUBioCompute/TopoPharmDTI/blob/main/Figure/Case%20study.png)
