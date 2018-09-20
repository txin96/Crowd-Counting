# coding=utf-8
import xml.etree.ElementTree as ET
import json
import os
import cv2

ANNOTATION_PATH = '../baidu_star_2018_train_stage2/annotation/annotation_train_stage2.json'
IMG_PATH = '../baidu_star_2018_train_stage2/image'

def create_xml(save_path, image_info):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 创建根节点
    root_xml = ET.Element("annotation")
    # 创建子节点，并添加属性
    folder = ET.SubElement(root_xml, "folder")
    folder.text = 'train'
    # 创建子节点，并添加数据
    filename = ET.SubElement(root_xml, "filename")
    filename.text = image_info['name'][13:]
    # Source
    source = ET.SubElement(root_xml, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    # Size
    size = ET.SubElement(root_xml, "size")
    img = cv2.imread(os.path.join(IMG_PATH, image_info['name']))
    width = ET.SubElement(size, "width")
    width.text = str(img.shape[1])
    height = ET.SubElement(size, "height")
    height.text = str(img.shape[0])
    depth = ET.SubElement(size, "depth")
    depth.text = str(img.shape[2])
    # Segmented
    segmented = ET.SubElement(root_xml, "segmented")
    segmented.text = "0"
    bbox = image_info['annotation']

    for i in bbox:
        # Object
        object_xml = ET.SubElement(root_xml, "object")
        # Name
        obj_name = ET.SubElement(object_xml, "name")
        obj_name.text = "person"
        # Pose
        pose = ET.SubElement(object_xml, "pose")
        pose.text = "Unspecified"
        # truncated
        truncated = ET.SubElement(object_xml, "truncated")
        truncated.text = "0"
        # difficult
        difficult = ET.SubElement(object_xml, "difficult")
        difficult.text = "0"
        # Bounding box
        bndbox = ET.SubElement(object_xml, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(i['x'])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(i['y'])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(i['x'] + i['w'])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(i['y'] + i['h'])
    # 创建elementtree对象，写文件
    tree = ET.ElementTree(root_xml)
    tree.write(os.path.join(save_path, image_info['name'][13:-4] + ".xml"))


annotations_file = json.load(open(ANNOTATION_PATH))
annotations = annotations_file['annotations']
for i in range(len(annotations)):
    ann = annotations[i]
    if ann['type'] == 'bbox':
        create_xml('img_xml', ann)
