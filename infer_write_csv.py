# encoding=utf-8

import os
import cv2
import numpy as np
import paddle.fluid as fluid
import paddle.v2 as paddle
import json
import csv
import math
import matplotlib.pyplot as plt


MODEL_PATH = 'model_infer/179'
TEST_IMAGE_PATH = 'baidu_star_2018_test_stage2/baidu_star_2018/image/stage2/test'
ANNOTATION_PATH = 'baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json'

CSV_PATH = 'result_resnet.csv'


def train_read(image_path, annotation_path):
    annotation_file = json.load(open(annotation_path))
    annotations = annotation_file['annotations']

    def reader():
        for i in range(len(annotations)):
            annotation = annotations[i]
            image_id = annotation['id']
            file_name = annotation['name'][12:]
            ignore_region = annotation['ignore_region']
            im = cv2.imread(os.path.join(image_path, file_name))
            if im is None:
                continue
            if len(ignore_region) > 0:
                for ir in ignore_region:
                    formatted_region = []
                    for p in ir:
                        formatted_region.append((p['x'], p['y']))
                    cv2.fillPoly(img=im, pts=[np.array(formatted_region)], color=0)
                    
            height = 16 * np.floor((im.shape[0]) / 16).astype(np.int64)
            width = 16 * np.floor((im.shape[1]) / 16).astype(np.int64)
            
            
            if width == 2048:
                width = 1632
                height = 1632
            
            if width > 1024:
                width = 16 * np.floor((im.shape[1]/2) / 16).astype(np.int64)
                height = 16 * np.floor((im.shape[0]/2) / 16).astype(np.int64)
            
            im = cv2.resize(im, (width, height))
            im = paddle.image.to_chw(im)

            yield np.array(im), np.array(image_id)

    return reader


def test(use_cuda, csv_file, BATCH_SIZE=1, model_dir='./model_infer'):
    out = open(csv_file, 'w')
    csv_writer = csv.writer(out, dialect='excel')
    csv_writer.writerow(['id', 'predicted'])
    # 数据定义
    image_shape = [3, 1024, 1024]
    label_shape = [1]
    # 输入数据
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 标签，即密度图
    label = fluid.layers.data(name='label', shape=label_shape, dtype='float32')

    # 获取测试数据
    test_reader = paddle.batch(
        train_read(TEST_IMAGE_PATH, ANNOTATION_PATH), batch_size=BATCH_SIZE)

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

        for test_data in test_reader():
            test_feat = np.array([data[0] for data in test_data]).astype(np.float32)
            test_id = np.array([data[1] for data in test_data]).astype(np.int32)
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: np.array(test_feat)},
                              fetch_list=fetch_targets)
            predict = np.sum(results[0], axis=(1, 2, 3))
            
            print "image_id: ", test_id[0], ", infer_results: ", int(math.floor(predict[0]+0.5))
            csv_writer.writerow([test_id[0], int(math.floor(predict[0]+0.5))])
            # test_data = test_reader().next()
            # test_feat = np.array([data[0] for data in test_data]).astype("float32")
            # test_label = numpy.array([data[1] for data in test_data]).astype("float32")

if __name__ == '__main__':
    test(use_cuda=True, csv_file=CSV_PATH, BATCH_SIZE=1, model_dir=MODEL_PATH)
