## YOLO-v2

------

## Introduction

Here is my experimental code implementation of the model described in the paper [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by pytorch.

The purpose of this code is to understand the entire implementation process of yolo-v2 step by step, especially how yolo-layer works. Therefore, the code is not very elegant.


## train

voc data format

    imgs
        1.jpg
        2.jpg
    xmls
        1.xml
        2.xml

* Set the target categories to be detected in tools/gen_anchor.py and train.py, respectively
* Specify the number of anchors and file paths in tools/gen_anchor.py, and run this python file to get anchors
* Copy the anchors to parser.--anchors in train.py, and specify file path and other parameters. Then python train.py

## detection

* Set the target categories to be detected in detect.py
* Copy the anchors to parser.--anchors in detect.py, and specify test image/images path, then python detect.py


## test result

<div align="center"><img src="result/2008_000803.jpg"></div>
<div align="center"><img src="result/2008_002631.jpg"></div>

## Due to the lack of equipment and large data sets, i only trained and tested on a part of the VOC 2007 data, and the results were not very good. I must say that training is difficult. So the purpose of this code is just to understand the implementation of yolo-v2.Here is my blog to explain how to build yolo-v2 step by step.[link](XXX)
