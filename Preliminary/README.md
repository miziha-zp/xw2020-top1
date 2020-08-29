# 队伍名称
MTM

# 方案说明
方案的预处理部分参考了https://github.com/ycd2016/xw2020_cnn_baseline/blob/master/baseline.py

因此在数据预处理部分代码有部分相似

NN包括9个模型，根据模型结构的不同，单个模型的训练时间在2h-10h不等（单卡RTX2080），由于不了解主办方的机器设置，在此串行运行9个模型，生成的概率文件将会stacking成为最终的文件。预计复现时间需要4天左右

# 训练说明
1. 基于CNN2D的深度学习模型，使用GTX2080Ti单卡五折交叉训练
2. 基于LightGBM的模型融合

# 运行说明
1. Docker运行后将会直接启动训练+融合过程，代码推荐使用nvidia docker运行(**测试通过**)，由于机器限制，我们仅在单卡RTX2080上进行了单卡的串行验证。
具体运行顺序如下：
```bash
docker build -t team_mtm_docker ./              
docker run --gpus '"device=0"' team_mtm_docker    # 启动gpu训练
``` 

# 文件夹结构

|--
    |-- Dockerfile  # Docker文件，将会运行当前目录下两个py文件并生成两个sub文件
    |-- data   # data
        |-- sensor_test.csv
        |-- sensor_train.csv
        |-- 提交结果示例.csv
    |-- PKL # 对应的数据文件（训练代码、模型及概率）
    |-- sub # 输出的预测文件，注意有两个文件
        |-- ensemble_1_to_allin088681.py      # 对应提交文件为 allin0.88681.csv	0.799
        |-- ensemble_2_to_0806allin088716.py  # 对应提交文件为 0806allin0.88716.csv 0.7979365079365079
    |-- lgb_online073.ipynb  使用Rolling的LightGBM实现，初赛线上0.73~，实际初赛融合未使用，仅供参考

注：
1.由于最后两次不能确定哪个对应榜单最高成绩，故都进行了复现。
2.**由于lgb多进程训练的原因，不能保证在不同配置的机器上的结果完全一致。** 

