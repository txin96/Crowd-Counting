#encoding=utf-8

import os
import cv2
import numpy as np
import paddle.fluid as fluid
import paddle.v2 as paddle

OUTPUT_PATH = 'training_data_stage2_256'
MODEL_PATH = 'model_infer/179'

IMAGE_PATH = 'training_data_stage2_256'
TEST_IMAGE_PATH = os.path.join(IMAGE_PATH, 'val')
TEST_DENSITY_PATH = os.path.join(IMAGE_PATH, 'val_den')


def train_read(train_image_path, train_label_path):
    datafile = os.listdir(train_image_path)
    def reader():
        for file_name in datafile:
        	im = cv2.imread(os.path.join(train_image_path, file_name))
        	# print file_name
                height = 16 * np.floor(im.shape[0] / 16).astype(np.int64)
                width = 16 * np.floor(im.shape[1] / 16).astype(np.int64)
		im = cv2.resize(im, (width, height))
		im = paddle.image.to_chw(im)
        	my_matrix = np.loadtxt(os.path.join(train_label_path, file_name.replace('.jpg', '.csv')), delimiter=",", skiprows=0)
		yield np.array(im), np.array(my_matrix)

    return reader


def test(use_cuda, BATCH_SIZE=4, model_dir='./model'):
    # 数据定义
    image_shape = [3, 1024, 1024]
    label_shape = [1, 1024, 1024]
    # 输入数据
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 标签，即密度图
    label = fluid.layers.data(name='label', shape=label_shape, dtype='float32')

    # 获取测试数据
    test_reader = paddle.batch(
        train_read(TEST_IMAGE_PATH, TEST_DENSITY_PATH), batch_size=BATCH_SIZE)

    # 是否使用GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 创建调试器
    exe = fluid.Executor(place)
    # 指定数据和label的对于关系
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    # 初始化调试器
    exe.run(fluid.default_startup_program())

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(model_dir, exe))
        # test_reader = paddle.batch(test_reader, batch_size=BATCH_SIZE)
	error_rate = []
        for test_data in test_reader():
            test_feat = np.array([data[0] for data in test_data]).astype(np.float32)            
	    test_label = np.array([data[1] for data in test_data]).astype(np.float32)
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: np.array(test_feat)},
                              fetch_list=fetch_targets)
	    predict = np.sum(results[0])
	    truth = np.sum(test_label)
            print "infer_results: ", predict, ", ground_truths: ", truth
	    error_rate.append(abs(predict-truth)/(truth))
	print np.mean(error_rate)
        # test_data = test_reader().next()
        # test_feat = np.array([data[0] for data in test_data]).astype("float32")
        # test_label = numpy.array([data[1] for data in test_data]).astype("float32")

        # print("ground truth: ", test_label)


if __name__ == '__main__':
    test(use_cuda=True, BATCH_SIZE=1, model_dir=MODEL_PATH)
