# SHN-based-2D-Face-Alignment-Algorithms

Stacked Hourglass Network for 2D face alignment

This ia an pytorch implemention for face alignment with stacked hourglass network (SHN). 

Pytorch >= 0.4

Python3

## Training and testing 
* In our experiments, we used stack=2 input resolution=256
```sh
python main.py 
```
* Run evaluation to get val score.
```sh
python main.py --phase test
```
