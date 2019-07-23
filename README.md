# Stacked Hourglass Network for 2D face alignment

This ia a PyTorch implemention for face alignment with stacked hourglass network (SHN). We use the normalized mean errors (NME), cumulative errors distribution (CED) curve, area under the curve (AUC), and failure rate to measure the landmark location error. This code (SHN-based) have achieved outstanding performance on 300-W and WFLW datasets. 

<div align=center><img src="https://github.com/face-alignment-group-of-ahucs/SHN-based-2D-face-alignment/blob/master/2376images.jpg" width="150" height="225" /><img src="https://github.com/face-alignment-group-of-ahucs/SHN-based-2D-face-alignment/blob/master/18--images.jpg" width="150" height="225" /><img src="https://github.com/face-alignment-group-of-ahucs/SHN-based-2D-face-alignment/blob/master/13--images.jpg" width="150" height="225" /><img src="https://github.com/face-alignment-group-of-ahucs/SHN-based-2D-face-alignment/blob/master/856--images.jpg" width="150" height="225" /><img src="https://github.com/face-alignment-group-of-ahucs/SHN-based-2D-face-alignment/blob/master/91--images.jpg" width="150" height="225" /></div>

## Performance

### 300W

| NME | *common*| *challenge* | *full* | *test*|
|:--:|:--:|:--:|:--:|:--:|
|2-HG-flip | 4.0 | - | - | - |

### WFLW

| NME |  *test* | *pose* | *illumination* | *occlution* | *blur* | *makeup* | *expression* |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|2-HG | 5.41 | 10.03 | 5.56 | 5.54 | 6.03 | 7.00 | 6.25 |

## Install

* `Python 3`

* `Install Pytorch >= 0.4 following the [official instructions](https://pytorch.org/)`

## data

You need to download images (e.g., 300W) from official websites and then put them into `data` folder for each dataset.

Your `data` directory should look like this:

````
SHN-based-2D-face-alignment
-- data
   |-- 300w
   |   |-- afw
   |   |-- helen
   |   |-- ibug
   |   |-- lfpw
````  

## Training and testing 
* For training, stacks = 1, input resolution = 256 
```sh
python main.py 
```
* Run evaluation to get result.
```sh
python main.py --phase test
```
