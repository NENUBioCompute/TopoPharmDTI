# TopoPharmDTI
![image](https://github.com/NENUBioCompute/TopoPharmDTI/blob/main/Figure/Model%20architecture.png)
# Resources:

+ result file: benchmark dataset, label reversal dataset, time-split dataset

##  Source codes:
+ cpi_model.py: TopoPharmDTI model file
+ smiles2graph.py: Generate drug features
+ smiles2subgraph.py: Generate drug subgraph features
+ protein_embedding.py: Generate protein features
+ train.py: train a TopoPharmDTI model.


## Trained models
Trained models and data is now available freely at https://drive.google.com/drive/folders/1IsCny9cWGYaemmvuGNDfMbvHkNrfaHBa?usp=sharing.

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
