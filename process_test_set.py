import os
import json
import numpy as np
import cv2

DATASET_PATH = 'baidu_star_2018_test_stage2/baidu_star_2018/image/stage2/test'
ANNOTATION_PATH = 'baidu_star_2018_test_stage2/baidu_star_2018/annotation/annotation_test_stage2.json'
OUTPUT_PATH = 'baidu_star_2018_test_stage2_processed'
IMAGE_SAVE_PATH = os.path.join(OUTPUT_PATH, 'image')


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)


def get_annotation(point_dict):
    pd = point_dict['annotation']
    point_list = []
    for pt in pd:
        point_list.append([pt['x'], pt['y']])

    return np.array(point_list)


num_images = 2859
indices = np.random.permutation(num_images)
annotations = json.load(open(ANNOTATION_PATH))
for index in range(num_images):
    i = indices[index]

    if index % 50 == 0:
        print 'Processing', index + 1, '/', num_images, 'files'

    annotation = annotations['annotations'][i]
    input_img_name = annotation['name'][12:]
    # print input_img_name

    ignore_region = annotation['ignore_region']

    
    im_colored = cv2.imread(os.path.join(DATASET_PATH, input_img_name))
    
    if len(ignore_region) > 0:
        for ir in ignore_region:
            formatted_region = []
            for p in ir:
                formatted_region.append((p['x'], p['y']))
            cv2.fillPoly(img=im_colored, pts=[np.array(formatted_region)], color=0)
    
    cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, input_img_name), img=im_colored)

