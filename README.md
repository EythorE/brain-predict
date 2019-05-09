# brain-predict
Valið efni í rafmagns- og tölvuverkfræði (RAF609M vor 2019)

## Description:
This is an implementation of a 3D-CNN network used to predict sex and age from T1w MRIs

###File:
Conv3D.ipynb        - Jupyter notebook with CNN implemention 
plot_results.ipynb  - Jupyter notebook for visualizing results
parameter_search.py - Performs hyper-parameter with Hyperopt
read_results.py     - Shell script for reading .csv with parameters and results from parameter\_search.py
utils.py            - Various functions used in the code
conv3D-weights.h5   - Weights for a pre-trained network 
dropout30epochs_results.csv - Results and training data from parameter search over 30 epochs

###Folder:
data                - Contains preprocessed data and labels
data_preprocess     - Contains various code used to preprocess the data


