# JNeRF-HSR

本项目为使用计图框架实现的 [NeRF-HSR](https://arxiv.org/abs/2304.08706)


## Dataset
 - 合成数据集 [Google Drive](https://drive.google.com/file/d/11tdooelweg4qzsYN1VzX8zw0-CXO4t9L/view?usp=share_link)（仅包含 RGB 图片），相机参数可以从 [DTU Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) 中获取
 - 真实数据集 [Google Drive](https://drive.google.com/file/d/1ULYdffLJIRVLWAeaRGqhALS1FbiOpf77/view?usp=share_link)
 - 真实数据集 [百度网盘](https://pan.baidu.com/s/1ED0qKmvSr2h5Il15IJ50jA?pwd=1234) 解压`dataset.zip`文件，其中`dataset1`文件夹是数据集

## Comparison

![](./gifs/buddha.gif)
![](./gifs/plate.gif)
![](./gifs/bronze.gif)

### Novel View Synthesis

![](./gifs/novel.gif)

### Mirror case

![](./gifs/mirror.gif)

## Running

1. 修改`JNeRF-HSR/confs/womask.conf`文件，将数据集和存储路径改为本地路径。
2. 运行 `train.sh` 使用我们提供的默认参数来运行代码。

## Acknowlegement

本项目基于 [NeuS](https://github.com/Totoro97/NeuS), [Jittor](https://github.com/Jittor/jittor), [NeRF-HSR](https://github.com/JiaxiongQ/NeuS-HSR) 的代码
感谢这些优秀的开源项目。


# Citation 

如果这个仓库对您有帮助，请引用原作者的工作
```
@InProceedings{Qiu_2023_CVPR,
author = {Qiu, Jiaxiong and Jiang, Peng-Tao and Zhu, Yifan and Yin, Ze-Xin and Cheng, Ming-Ming and Ren, Bo},
title = {Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
```

## 常见问题

jittor 框架在进行并行矩阵相乘运算时使用GPU加速会检测矩阵的前 n 维，导致报错，但 CPU 模式不会出问题。

- 解决办法：
在本地jittor库里修改代码： `jittor/nn.py`文件中注释`120~124`行，让矩阵相乘不进行GPU加速。
