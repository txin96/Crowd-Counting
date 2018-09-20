# encoding=utf-8
import os
import cv2
import numpy as np
import paddle.fluid as fluid
import paddle.v2 as paddle
from model import res_net

OUTPUT_PATH = './training_data_stage2_256'
TRAIN_IMAGE_PATH = os.path.join(OUTPUT_PATH, 'train')
TRAIN_DENSITY_PATH = os.path.join(OUTPUT_PATH, 'train_den')
VAL_IMAGE_PATH = os.path.join(OUTPUT_PATH, 'val')
VAL_DENSITY_PATH = os.path.join(OUTPUT_PATH, 'val_den')


def train_read(train_image_path, train_label_path):
    datafile = os.listdir(train_image_path)
    train_image = []
    train_label = []
    train_people = []
    # k = 0
    print 'Start reading data...'
    for file_name in datafile:
        #if k > 10:
        #    break
        #k+=1

        im = cv2.imread(os.path.join(train_image_path, file_name))
        my_matrix = np.loadtxt(os.path.join(train_label_path, file_name.replace('.jpg', '.csv')), delimiter=",",
                               skiprows=0)
        train_people.append(np.sum(my_matrix))
        my_matrix = cv2.resize(my_matrix, (im.shape[1]/8, im.shape[0]/8))
        im = paddle.image.to_chw(im)
        
        train_image.append(im)
        my_matrix = my_matrix * 64
        train_label.append(my_matrix)
        

    train_image = np.array(train_image)
    train_label = np.array(train_label)
    train_people = np.array(train_people)
    print 'Finish reading data...'

    def reader():
        for i in xrange(train_image.shape[0]):
            yield train_image[i], train_label[i], train_people[i]

    return reader


def train(use_cuda, learning_rate, num_passes, BATCH_SIZE=1, model_save_dir='model', inference_dir='/model_infer'):
    # 数据定义
    image_shape = [3, 256, 256]
    label_shape = [32, 32]
    # 输入数据
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 标签，即密度图
    label = fluid.layers.data(name='label', shape=label_shape, dtype='float32')

    people_num = fluid.layers.data(name='people', shape=[1], dtype='float32')

    # 获取神经网络的输出
    predict = res_net(image)
    predict_people = fluid.layers.reduce_sum(predict, dim=[2, 3])
    density_loss = fluid.layers.mean(fluid.layers.square_error_cost(predict, label))
    people_loss = fluid.layers.mean(fluid.layers.elementwise_div(fluid.layers.square_error_cost(predict_people, people_num), fluid.layers.square(people_num+1)))
    cost = density_loss + 0.1 * people_loss

    # 定义优化方法
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[150000, 290000],
            values=[0.000001, 0.0000001, 0.00000001]))
    '''
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[1000000, 2200000],
            values=[0.000001, 0.0000001, 0.00000001]))
    
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=learning_rate,
            decay_steps=950000,
            decay_rate=0.1,
            staircase=True))
    '''

    opts = optimizer.minimize(cost)
    # 获取训练数据
    train_reader = paddle.batch(
        train_read(TRAIN_IMAGE_PATH, TRAIN_DENSITY_PATH), batch_size=BATCH_SIZE)

    # 是否使用GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 创建调试器
    exe = fluid.Executor(place)

    # 指定数据和label的对于关系
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label, people_num])

    # 初始化调试器
    exe.run(fluid.default_startup_program())

    # fluid.io.load_params(executor=exe, dirname='stage2/model_fine_tune/299')
    # 开始训练，使用循环的方式来指定训多少个Pass
    for pass_id in range(num_passes):
        # 从训练数据中按照一个个batch来读取数据
        loss_list = []
        for batch_id, data in enumerate(train_reader()):
            loss, = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[cost])
            loss_list.append(loss)
            # accuracy.add(value=acc, weight=weight)
            # if batch_id % 20 == 0:
            # print("Pass {0}, batch {1}, loss {2}".format(pass_id, batch_id, loss[0]))
            # print("Pass {0}, batch {1}, loss {2}, acc {3}".format(pass_id, batch_id, loss[0], 0))
        print("Pass {0},  loss {1}".format(pass_id, np.mean(loss_list)))

        # 指定保存模型的路径
        model_path = os.path.join(model_save_dir, str(pass_id))
        inference_model_path = os.path.join(inference_dir, str(pass_id))
        # 如果保存路径不存在就创建
        # 每20个Pass保存一次预测的模型
        if pass_id > 0 and pass_id % 20 == 0:
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if not os.path.exists(inference_model_path):
                os.makedirs(inference_model_path)
            print 'save model to %s' % (model_path)
            fluid.io.save_params(executor=exe, dirname=model_path, main_program=None)
            print 'save inference model to %s' % (inference_model_path)
            fluid.io.save_inference_model(dirname=inference_model_path, feeded_var_names=['image'],
                                          target_vars=[predict], executor=exe)
        # 训练结束后保存一次模型
        if pass_id == num_passes - 1:
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if not os.path.exists(inference_model_path):
                os.makedirs(inference_model_path)
            print 'save model to %s' % (model_path)
            fluid.io.save_params(executor=exe, dirname=model_path, main_program=None)
            print 'save inference model to %s' % (inference_model_path)
            fluid.io.save_inference_model(dirname=inference_model_path, feeded_var_names=['image'],
                                          target_vars=[predict], executor=exe)


if __name__ == '__main__':
    # 开始训练
    train(use_cuda=True, learning_rate=0.000001, num_passes=300, BATCH_SIZE=1, model_save_dir='model')
