# CSI-Based-Activity-Recognition

### data preprocess
使用matlab进行数据预处理：
- 对幅度（mag）进行滤波
- 对相位（phase）进行接缠

### data load
使用python进行数据加载，用于之后的神经网络训练：
运行train.py即可加载数据，关键参数请在train.py中修改

文件夹结构：

dataload

   --> PE(文件夹存放原始数据)
   
   --> dataprocess.py
   
   --> train.py
