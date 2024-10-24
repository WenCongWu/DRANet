### 1 dataset download

Download the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php), and put these datasets into ./Datasets/ directory. You can also download this dataset from Google Drive [[train set](https://drive.google.com/file/d/1s8gNDFjYOcD6rIj9N2G-niQkQ-2e9KJn/view?usp=drive_link), [val set](https://drive.google.com/file/d/1caJDtXwZ38Mq8pMvEm7-vEyEHay0hjVj/view?usp=drive_link), [test set](https://drive.google.com/file/d/1r5WlKYBUUJGHEsq6_-xybxfXrAnA2QGB/view?usp=drive_link)].

### 2 train

- Generate image patches from high-resolution training images of SIDD dataset

```
python generate_patches_SIDD.py --ps 128 --num_patches 300 --num_cores 16
```

- train the model with default arguments by running

```
python train.py
```

### 3 test

- download the [DND](https://noise.visinf.tu-darmstadt.de/), and put these datasets into ./Datasets/ directory.

### testing on SIDD dataset

```
python test_SIDD.py
```

### testing on DND dataset

```
python test_DND.py
```

### in order to get the PSNR and SSIM values of the denoised SIDD test set, run MATLAB script
```
evaluate_SIDD.m
```
