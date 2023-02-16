# Domain Adaptation in Segmenting Historical Maps: a Weakly Supervised Approach through Spatial Co-occurrence

### Introduction
This repository contains source code (Keras + tensorflow implementation) for the [paper](https://www.sciencedirect.com/science/article/pii/S0924271623000278?utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_medium=email&utm_acid=210999134&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&CMX_ID=&utm_in=DM341514&utm_source=AC_) by Sidi Wu, Konrad Schindler, Magnus Heitzler and Lorenz Hurni. 

### Network architecture
![image](https://user-images.githubusercontent.com/36080548/219400588-7b61e43c-81d7-40d9-95b8-0567f3d0a8e1.png)
The proposed network contains three parts:
* Segmentation model 
* Co-occurrrence detection model (detect co-occurring objects between source and target and provide weak supervision to the target)
* Discriminator (on the entropy of the prediction map)

### Explaination of scripts
* train.py - to train the segmentation model + co-occurrence detection model
* trainGAN.py - to train the segmentation model + co-occurrence detection model + discriminator (adversarial training)
* demo.py - to visualize results
* prediction.py - to generate predictions given a map sheet 

### Datasamples
Training data samples are located in "Datasamples", which can be categorized into "labelled", "unlabelled" and "paired" data. All data is stored as ".npz".

An [old national map sheet](https://www.polybox.ethz.ch/index.php/s/sis7JpXflRBi9jy) of 14000 * 9600 pixels is uploaded in Polybox to generate predictions for the whole map sheet.

Due to copyright issue, not all training/testing data can be publicly provided. Please send request to sidiwu@ethz.ch for further discussion.

### Citation
If you want to use the scripts, please cite the paper:
@article{WU2023199,
title = {Domain adaptation in segmenting historical maps: A weakly supervised approach through spatial co-occurrence},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {197},
pages = {199-211},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.01.021},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623000278},
author = {Sidi Wu and Konrad Schindler and Magnus Heitzler and Lorenz Hurni}
}

