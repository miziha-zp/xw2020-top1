# 指定python版本，这里使用3.6


FROM registry.hub.docker.com/tensorflow/tensorflow:2.1.0-gpu-py3

# 安装需要的python包，国内指定某些源，下载比较快，此处使用清华源（可以指定你用的版本，如tensorflow=1.0）

RUN pip install numpy pandas scikit-learn lightgbm seaborn tqdm scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./data/ xx/data    
COPY ./ensemble_1_to_allin088681.py xx/
COPY ./ensemble_2_to_0806allin088716.py xx/
COPY ./PKL/ xx/PKL
COPY ./sub/ xx/sub
# 如果你有多个目录，也需要依次拷贝到docker 容器中
# 使用/* 的格式在docker中可能会造成错误
# 设置工作目录为xx

WORKDIR xx

# 下面是测试
#CMD echo start | tee start_ && pip list && nvidia-smi && python PKL/test_gpu.py && bash PKL/run.sh && echo done |tee done_ && sleep 2h


# 生成所需的最终文件的时间大概为4天（RTX2080 GPU,共9个五折模型）
CMD echo start | tee start_ && pip list && nvidia-smi && python PKL/test_gpu.py && bash PKL/run.sh && python ensemble_2_to_0806allin088716.py && python ensemble_1_to_allin088681.py && echo done |tee done_ && sleep 2h

