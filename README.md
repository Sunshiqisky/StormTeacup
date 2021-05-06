# StormTeacup

This repository contains the pytorch code for the article: A Strom in a Teacup -- A Lung Microphysiological System with a Deep-Learning Algorithm to Monitor Lung Pathological and Inflammatory Reactions.

### 1、Getting Started

```
git clone https://github.com/Sunshiqisky/StormTeacup.git
```

### 2、Requirements

```
python>=3.6
opencv-python (4.2.0)
torch (1.2.0)
torchvision (0.4.0)
tensorboardx
matplotlib
numpy
pillow
```

### 3、Run the file

**preprocessing.py**：Preprocessing the input image with preprocessing.py is mainly to remove the influence of the lower right ruler.

**main.py**：You can place training sets and validation sets in the data folder,then run the main.py to start the trainning.

**imageclassify.py**：img_path = "", you can replace the image path for your image.The output for the result.numpy(), a value of 0/1 means that the image is judged as the Control group /LPS group.

