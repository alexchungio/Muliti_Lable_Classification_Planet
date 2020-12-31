# Muliti_Lable_Classification_Planet
Try to use multi-label-classification realize Planet-Kaggle task


## Dataset

* <https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data>
* <https://www.kaggle.com/nikitarom/planets-dataset>

## Train

### change config 

```shell script
vim ./configs/cfgs.py
```

## Preprocess

### Count dataset
```shell script
python ./tools/preprocess.py
```

###  step2: Get mean and std

```shell script
python ./tools/get_mean_std.py
```

## Train
```shell script
python ./train.py
```
### Train log

* training
<p align=center>
  <img src="docs/graph/train.png" width="1332" height="446">
</p>
* epoch
<p align=center>
  <img src="docs/graph/epoch.png" width="1332" height="446">
</p>
* learning rate
<p align=center>
  <img src="docs/graph/lr.png" width="491" height="429">
</p>

## Inference

```shell script
python ./inference.py
```

## Bag of Tricks

* Multi-fold cross-validation

* Get best threshold  with brute force search

* Stop when loss plateaus

* Image Augmentation

* ...

## TODO


## Reference 
* <https://github.com/rwightman/pytorch-planet-amazon>

