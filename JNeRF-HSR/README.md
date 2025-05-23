# NeuS-HSR
This is the official repo for the implementation of "[Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections](https://openaccess.thecvf.com/content/CVPR2023/html/Qiu_Looking_Through_the_Glass_Neural_Surface_Reconstruction_Against_High_Specular_CVPR_2023_paper.html)" paper (CVPR 2023). [Arxiv](https://arxiv.org/abs/2304.08706) [Video](https://www.youtube.com/watch?v=lwHd-GJAmMA)

## Synthetic Dataset
[Google Drive](https://drive.google.com/file/d/11tdooelweg4qzsYN1VzX8zw0-CXO4t9L/view?usp=share_link) only contains RGB images, camera parameters are from [DTU Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36).

## Real-world Dataset
[Google Drive](https://drive.google.com/file/d/1ULYdffLJIRVLWAeaRGqhALS1FbiOpf77/view?usp=share_link)

## Comparison

![](./gifs/buddha.gif)
![](./gifs/plate.gif)
![](./gifs/bronze.gif)

## Novel View Synthesis

![](./gifs/novel.gif)

## Mirror case

![](./gifs/mirror.gif)

## Acknowlegement
Our code is built on [NeuS](https://github.com/Totoro97/NeuS) project. 

Our real-world dataset is constructed by following [NeRF++](https://github.com/Kai-46/nerfplusplus) project. 

Thanks for these great projects!


## Citation 
If you use our code or method in your work, please cite the following:
```
@InProceedings{Qiu_2023_CVPR,
author = {Qiu, Jiaxiong and Jiang, Peng-Tao and Zhu, Yifan and Yin, Ze-Xin and Cheng, Ming-Ming and Ren, Bo},
title = {Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2023}
}
```

