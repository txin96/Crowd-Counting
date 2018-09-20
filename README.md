# Crowd-Counting
[2018百度之星开发者大赛](http://star.baidu.com/developer.html)总决赛优胜奖作品
## ResNet-Based Model

- 基于ResNet-18，使用彩色图像做输入，密度图做输出
- 密度图的生成参考了 Zhang et al. 的论文以及https://github.com/svishwa/crowdcount-mcnn/tree/master/data_preparation中的.m文件代码
- 训练时仅使用annotation type为dot的图片
- 模型经过数百万次迭代后达到了0.33752的error_rate

### Usage

1. 运行create_training_data.py对训练集图像进行分割，可设置IMG_SIZE使得分割图片大小统一
2. 运行train.py训练模型，初始learning_rate为0.000001， Optimizer选择Adam，batch_size设为1，总迭代次数约240万次
3. 运行infer.csv可对模型进行评估，infer_write_csv.py可生成符合提交条件的csv文件

## Object Detection(SSD)

- 适用于对人数不太多的图片做人数预测
- 训练时仅使用annotation type为 bbox 的图片
- 模型经过训练后达到了0.34954的error_rate
- 代码使用了PaddlePaddle的开源代码，参考 https://github.com/PaddlePaddle/models/tree/develop/ssd

### Usage

1. 运行create_xml.py生成适用于SSD训练的输入
2. 修改配置文件里的CLASS_NUM为2，BATCH_SIZE设为16，IMG_WIDTH和IMG_HEIGHT设置为512，LEARNING_RATE_DECAY_B设置为16551，适应迭代次数
3. 运行train.py训练模型
4. 修改infer.py中输出结果的方式，输出infer.res的同时输出包含文件名和人数的people_num.txt
5. 运行infer.py预测图片中的人数
6. 运行txt_to_csv.py将第5步得到的txt文件转换为csv文件(包含img_name列，不符合提交条件)

## Statistics and Combination

1. 经训练集的统计，type为bbox的图像数量与type为dot的图像数量比约为4:6，其中用于SSD训练的图像中人数范围从1到92，适用于ResNet-Based-Model训练的图像中人数从1到2000+不等，但人数的分布大致相同，集中于40人以下。
2. 因为SSD与ResNet-Based-Model训练的图片范围不同，以至于在不同的场景下有各自的优势（比如SSD适用于人少的场景，ResNet-Based-Model适用于人很多的场景），在实际预测时需要适当组合，选择最合适的模型预测出的人数，所以我们在预测结果上使用线性回归，组合出最接近真实值的人数值。该方法的思想类似于决策树，对预测结果分类处理。我们尽量把条件设置成可以将图片数量二等分或多等分，使得每个分支图像数量尽量均匀，同时我们也以图像的大小作为依据之一。使用的工具为Excel的数据分析工具。
3. 我们在训练集上验证该方法，ResNet-Based-Model预测的结果error_rate约为0.40，经与SSD的组合后，error_rate下降至0.33.


## 如何复现本项目

1. 下载本项目
2. 使用训练好的模型，并将预测结果组合使用
   - ResNet Based Model:
     - 如果想自己训练模型，则运行train.py
     - 运行infer_write_csv.py文件预测，结果输出为result_resnet.csv
   - SSD
     - 将当前工作目录切换至ssd文件夹下
     - 如果想自己训练模型，则运行train.py
     - 预测前请先运行process_test_set.py将图片的ignore_region添加到图片上
     - 运行infer.py文件预测，结果输出为people_num.txt
     - 运行根目录下的txt_to_csv.py文件将people_num.txt转换为result_ssd.csv
   - 组合使用
     - 手动拼接(与自动拼接选择一个即可)
       - 将result_ssd.csv与result_resnet.csv横向拼接，第一列为id，第二列为ssd的预测结果，第三列为ResNet Based Model的预测结果，同时将第一行标题删除，并将文件命名为result_wait_to_process.csv
     - 自动拼接
       - 运行merge_csv.py
     - 运行concate_csv.py生成符合提交条件的result.csv