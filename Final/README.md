# 说明
1. 赛场提供的是Tensorflow2.0，使用Attention层有问题，因此从新版本Tensorflow上复制了[dense_attention.py](https://github.com/tensorflow/tensorflow/blob/498e815097e74aff7fefdbbae69ba9daf6e9c023/tensorflow/python/keras/layers/dense_attention.py#L191)文件，手动导入

# 文件结构
```
|--
    |-- code   # 复赛使用到的代码
        |-- CNNtower.ipynb         # 双塔CNN代码
        |-- cos_dense_attention.py    # 见说明1
        |-- lgb_final.ipynb           # LightGBM模型
        |-- LSTMnet.ipynb             # LSTM代码 
        |-- ensembl8086.ipynb         # 融合的LightGBM代码 
    |-- data # 由于主办方要求，无法提供数据文件
```