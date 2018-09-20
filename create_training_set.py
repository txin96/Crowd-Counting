import os
import json
import numpy as np
import cv2
from get_density_map_gaussian import get_density_map_gaussian

DATASET_PATH = 'baidu_star_2018_train_stage2/image'
ANNOTATION_PATH = 'baidu_star_2018_train_stage2/annotation/annotation_train_stage2.json'
OUTPUT_PATH = 'training_data_stage2_256'
TRAIN_IMAGE_PATH = os.path.join(OUTPUT_PATH, 'train')
TRAIN_DENSITY_PATH = os.path.join(OUTPUT_PATH, 'train_den')
VAL_IMAGE_PATH = os.path.join(OUTPUT_PATH, 'val')
VAL_DENSITY_PATH = os.path.join(OUTPUT_PATH, 'val_den')

IMAGE_SIZE = 256

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists(TRAIN_IMAGE_PATH):
    os.makedirs(TRAIN_IMAGE_PATH)
if not os.path.exists(TRAIN_DENSITY_PATH):
    os.makedirs(TRAIN_DENSITY_PATH)
if not os.path.exists(VAL_IMAGE_PATH):
    os.makedirs(VAL_IMAGE_PATH)
if not os.path.exists(VAL_DENSITY_PATH):
    os.makedirs(VAL_DENSITY_PATH)


def get_annotation(point_dict):
    pd = point_dict['annotation']
    point_list = []
    for pt in pd:
        point_list.append([pt['x'], pt['y']])

    return np.array(point_list)


num_images = 2859
num_val = 200
indices = np.random.permutation(num_images)
annotations = json.load(open(ANNOTATION_PATH))
val_num = 0
for index in range(num_images):
    i = indices[index]

    if index % 50 == 0:
        print 'Processing', index + 1, '/', num_images, 'files'

    annotation = annotations['annotations'][i]
    input_img_name = annotation['name']

    ann_type = annotation['type']
    if ann_type != 'dot':
        continue

    ignore_region = annotation['ignore_region']

    im = cv2.imread(os.path.join(DATASET_PATH, input_img_name), cv2.IMREAD_GRAYSCALE)
    im_colored = cv2.imread(os.path.join(DATASET_PATH, input_img_name))

    if len(ignore_region) > 0:
        for ir in ignore_region:
            formatted_region = []
            for p in ir:
                formatted_region.append((p['x'], p['y']))
            cv2.fillPoly(img=im_colored, pts=[np.array(formatted_region)], color=0)

    [h, w] = im.shape
    wn2 = w / 8
    hn2 = h / 8
    wn2 = 8 * np.floor(wn2 / 8)
    hn2 = 8 * np.floor(hn2 / 8)

    # annPoints = image_info{1}.location;
    annPoints = get_annotation(annotation)

    if w <= 2 * wn2:
        # im = imresize(im, [h, 2 * wn2 + 1])
        im = cv2.resize(im, (h, 2 * wn2 + 1))
        im_colored = cv2.resize(im_colored, (h, 2 * wn2 + 1))
        annPoints[:, 0] = annPoints[:, 0] * 2 * wn2 / w

    if h <= 2 * hn2:
        # im = imresize(im, [2 * hn2 + 1, w])
        im = cv2.resize(im, (2 * hn2 + 1, w))
        im_colored = cv2.resize(im_colored, (2 * hn2 + 1, w))
        annPoints[:, 1] = annPoints[:, 1] * 2 * hn2 / h

    [h, w] = im.shape
    a_w = wn2 + 1
    b_w = w - wn2
    a_h = hn2 + 1
    b_h = h - hn2
    im_density = get_density_map_gaussian(im, annPoints)

    if len(ignore_region) > 0:
        for ir in ignore_region:
            formatted_region = []
            for p in ir:
                formatted_region.append((p['x'], p['y']))
            cv2.fillPoly(img=im_density, pts=[np.array(formatted_region)], color=0)

    if val_num < num_val:
        cv2.imwrite(os.path.join(VAL_IMAGE_PATH, '%s%s' % (input_img_name[13:-4], '.jpg')),
                    img=im_colored)
        np.savetxt(os.path.join(VAL_DENSITY_PATH, '%s%s' % (input_img_name[13:-4], '.csv')),
                   im_density, delimiter=',')
        val_num+=1
        continue

    h_num = int(np.floor(h / IMAGE_SIZE))
    w_num = int(np.floor(w / IMAGE_SIZE))

    im_sampled_list = []
    im_density_sampled_list = []
    for k in range(h_num):
        for w in range(w_num):
            im_sampled = im_colored[IMAGE_SIZE * k: IMAGE_SIZE * (k + 1), IMAGE_SIZE * w:IMAGE_SIZE * (w + 1)]
            im_density_sampled = im_density[IMAGE_SIZE * k: IMAGE_SIZE * (k + 1), IMAGE_SIZE * w:IMAGE_SIZE * (w + 1)]
            x = im_density_sampled[np.where(im_density_sampled > 0)]
            if len(x) > 0.5:
                im_sampled_list.append(im_sampled)
                im_density_sampled_list.append(im_density_sampled)
    
    if len(im_sampled_list) > 4:
        image_list = np.random.choice(range(len(im_sampled_list)), max(4, int(len(im_sampled_list) / 1.5)))
    else:
        image_list = range(len(im_sampled_list))

    for j in range(len(image_list)):
        j = image_list[j]
        cv2.imwrite(os.path.join(TRAIN_IMAGE_PATH, '%s%s%s%s' % (input_img_name[13:-4], '_', str(j), '.jpg')),
                    img=im_sampled_list[j])
        np.savetxt(os.path.join(TRAIN_DENSITY_PATH, '%s%s%s%s' % (input_img_name[13:-4], '_', str(j), '.csv')),
                   im_density_sampled_list[j], delimiter=',')

