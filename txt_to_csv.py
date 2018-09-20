import json
import csv

ANNOTATION_PATH = 'annotation/annotation_test_stage2.json'

annotations_file = json.load(open(ANNOTATION_PATH))
annotations = annotations_file['annotations']

f = open('./people_num.txt')
out = open('./result_ssd.csv', 'w')
csv_writer = csv.writer(out, dialect='excel')
csv_writer.writerow(['id', 'predicted'])
for line in f:
    parts = line.split('\t')
    img_name = parts[0]
    number = parts[1]
    for i in range(len(annotations)):
        ann = annotations[i]
        img = ann['name'][12:]
        if img_name[62:] == img:
            csv_writer.writerow([ann['id'], int(number)])
            break
