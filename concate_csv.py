import cv2
import math
import json
import csv


def my_round(x):
    return math.floor(x + 0.5)


ANNOTATION_PATH = 'annotation/annotation_test_stage2.json'

IMG_PATH = 'stage2/test_processed/'


annotations_file = json.load(open(ANNOTATION_PATH))
annotations = annotations_file['annotations']

reader = csv.reader(open('result_wait_to_process.csv', 'r'))
out = open('./result.csv', 'w')
csv_writer = csv.writer(out, dialect='excel')
csv_writer.writerow(['id', 'predicted'])
# distribution_ssd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# distribution_resnet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for line in reader:
    line_id = int(line[0])
    ssd = int(line[1])
    resnet = int(line[2])
    img_name = line[3]
    if line_id % 50 == 0:
        print 'Processing No.', line_id + 1, 'files'

    number = 1
    if ssd > resnet:
        # distribution_ssd[ssd / 20] += 1
        if ssd >= 60:
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] == 1920:
                number = int(my_round(ssd * 0.8 + resnet * 0.1))
            else:
                number = int(my_round(ssd * 0.8 + resnet * 0.2))
        elif ssd >= 40:
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] == 1920:
                number = int(my_round(ssd * 0.9 + resnet * 0.1))
            else:
                if abs(float(ssd - resnet)) / ssd > 0.3 and abs(float(resnet - ssd)) / resnet < 1:
                    number = resnet
                else:
                    number = ssd
        elif ssd >= 20:
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] != 1920:
                number = int(my_round(ssd * 0.5 + resnet * 0.5))
            if resnet < 10:
                number = int(my_round(ssd * 0.8 + resnet * 0.2))
            else:
                if ssd >= 30:
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
                else:
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
        else:
            img = cv2.imread(IMG_PATH + img_name)
            if ssd >= 10:
                if resnet < 10:
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
                else:
                    if ssd - resnet >= 5:
                        if abs(float(ssd - resnet)) / ssd > 0.4:
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:
                            number = int(my_round(ssd * 0.8 + resnet * 0.2))
                    else:
                        if ssd - resnet >= 3:
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
            else:
                if resnet >= 5:
                    number = resnet
                else:
                    if abs(float(ssd - resnet)) / ssd >= 0.4:
                        if img.shape[1] == 1280:
                            number = resnet
                        else:
                            number = ssd
                    else:
                        if img.shape[1] == 960:
                            number = ssd
                        else:
                            number = resnet

    else:
        # distribution_resnet[resnet / 40] += 1
        if resnet >= 120:
            if resnet > 200:
                number = resnet
            else:
                number = int(my_round(ssd * 0.3 + resnet * 0.7))
        elif resnet >= 80:
            if resnet >= 100:
                number = int(my_round(ssd * 0.8 + resnet * 0.2))
            else:
                if (resnet - ssd) >= 40:
                    number = int(my_round(ssd * 0.3 + resnet * 0.7))
                else:
                    number = int(my_round(ssd * 0.5 + resnet * 0.5))
        elif resnet >= 40:
            number = int(my_round(ssd * 0.3 + resnet * 0.7))
        else:
            if resnet >= 20:
                number = int(my_round(ssd * 0.7 + resnet * 0.3))
            else:
                if resnet < 5:
                    if ssd == 0:
                        number = int(my_round(resnet * 0.7))
                    elif ssd == 1:
                        img = cv2.imread(IMG_PATH + img_name)
                        if img.shape[1] == 1920:
                            number = ssd
                        else:
                            number = resnet
                    else:
                        img = cv2.imread(IMG_PATH + img_name)
                        if img.shape[1] == 1920:
                            number = ssd
                        else:
                            number = resnet
                elif resnet == 5:
                    img = cv2.imread(IMG_PATH + img_name)
                    if ssd == 0:
                        number = resnet
                    else:
                        if img.shape[1] == 1920:
                            number = ssd
                        else:
                            number = resnet
                else:
                    img = cv2.imread(IMG_PATH + img_name)
                    if resnet < 10:
                        if img.shape[1] == 1920:
                            if abs(float(resnet - ssd)) / resnet > 0.5 and abs(float(resnet - ssd)) / ssd < 2:
                                number = resnet
                            elif 0.43 > abs(float(resnet - ssd)) / resnet > 0.4285:
                                number = ssd
                            else:
                                number = ssd
                        else:
                            number = ssd
                    else:
                        if resnet > 15:
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                            if ssd >= 10:
                                img = cv2.imread(IMG_PATH + img_name)
                                if img.shape[1] == 2048:
                                    number = ssd
                                else:
                                    number = int(my_round(ssd * 0.7 + resnet * 0.3))
                            else:
                                number = int(my_round(ssd * 0.5 + resnet * 0.5))

    csv_writer.writerow([line[0], number])

# print distribution_ssd
# print distribution_resnet
