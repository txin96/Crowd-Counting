import os
import cv2
import math
import json
import csv


def my_round(x):
    return math.floor(x + 0.5)


ANNOTATION_PATH = 'annotation/annotation_test_stage2.json'

IMG_PATH = 'stage2/test_processed/'
IMG_SAVE_PATH = 'see'

if not os.path.exists(IMG_SAVE_PATH):
    os.makedirs(IMG_SAVE_PATH)

annotations_file = json.load(open(ANNOTATION_PATH))
annotations = annotations_file['annotations']

reader = csv.reader(open('result_wait_to_process.csv', 'r'))
out = open('./result_concate.csv', 'w')
csv_writer = csv.writer(out, dialect='excel')
csv_writer.writerow(['id', 'predicted'])
img_sum = 0
distribution_ssd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
distribution_resnet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0]
for line in reader:
    line_id = int(line[0])
    ssd = int(line[1])
    resnet = int(line[2])
    img_name = line[3]

    number = 1
    if ssd > resnet:
        distribution_ssd[ssd / 20] += 1
        if ssd >= 60:  # 103
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] == 1920:  # 101
                number = int(my_round(ssd * 0.8 + resnet * 0.1))
            else:  # 2
                number = int(my_round(ssd * 0.8 + resnet * 0.2))
        elif ssd >= 40:  # 80
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] == 1920:  # 72
                number = int(my_round(ssd * 0.9 + resnet * 0.1))
            else:  # 8
                if abs(float(ssd - resnet)) / ssd > 0.3 and abs(float(resnet - ssd)) / resnet < 1:  # 2
                    number = resnet
                else:  # 6
                    number = ssd
        elif ssd >= 20:  # 577
            img = cv2.imread(IMG_PATH + img_name)
            if img.shape[1] != 1920:  # 12
                number = int(my_round(ssd * 0.5 + resnet * 0.5))
            if resnet < 10:  # 53
                # img_sum += 1
                # print line_id, ssd, resnet, img_name, abs(resnet - ssd), abs(
                #     float(ssd - resnet)) / ssd, abs(float(resnet - ssd)) / resnet, img.shape[1]
                # cv2.imwrite(os.path.join(IMG_SAVE_PATH, img_name), img)
                number = int(my_round(ssd * 0.8 + resnet * 0.2))
            else:  # 524
                if ssd >= 30:  # 233
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
                else:  # 291
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
        else:
            img = cv2.imread(IMG_PATH + img_name)
            if ssd >= 10:  # 470
                if resnet < 10:  # 244
                    number = int(my_round(ssd * 0.85 + resnet * 0.15))
                else:  # 226
                    if ssd - resnet >= 5:  # 72
                        if abs(float(ssd - resnet)) / ssd > 0.4:  # 18
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:  # 54
                            number = int(my_round(ssd * 0.8 + resnet * 0.2))
                    else:  # 154
                        if ssd - resnet >= 3:  # 64
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:  # 88
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
            else:  # 112
                if resnet >= 5:  # 51
                    number = resnet
                else:  # 61
                    if abs(float(ssd - resnet)) / ssd >= 0.4:  # 21
                        if img.shape[1] == 1280:
                            number = resnet
                        else:
                            number = ssd
                    else:  # 40
                        if img.shape[1] == 960:  # 19
                            number = ssd
                        else:  # 21
                            number = resnet

    else:
        distribution_resnet[resnet / 40] += 1
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
            else:  # 831
                if resnet < 5:  # 315
                    if ssd == 0:
                        number = int(my_round(resnet * 0.7))
                    elif ssd == 1:
                        img = cv2.imread(IMG_PATH + img_name)
                        if img.shape[1] == 1920:
                            number = ssd
                        else:
                            number = resnet
                    else:  # 186
                        img = cv2.imread(IMG_PATH + img_name)
                        if img.shape[1] == 1920:  # 27
                            number = ssd
                        else:  # 159
                            number = resnet
                elif resnet == 5:  # 43
                    img = cv2.imread(IMG_PATH + img_name)
                    if ssd == 0:  # 1
                        number = resnet
                    else:  # 31
                        if img.shape[1] == 1920:  # 31
                            number = ssd
                        else:  # 11
                            number = resnet
                else:  # 473
                    img = cv2.imread(IMG_PATH + img_name)
                    if resnet < 10:  # 62
                        if img.shape[1] == 1920:
                            if abs(float(resnet - ssd)) / resnet > 0.5 and abs(float(resnet - ssd)) / ssd < 2:  # 2
                                number = resnet
                            elif 0.43 > abs(float(resnet - ssd)) / resnet > 0.4285:  # 917 1165 change
                                number = ssd
                            else:  # 58
                                number = ssd
                        else:
                            number = ssd
                    else:
                        if resnet > 15:  # 211
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                        else:  # 200
                            number = int(my_round(ssd * 0.5 + resnet * 0.5))
                            if ssd >= 10:  # 116
                                img = cv2.imread(IMG_PATH + img_name)
                                if img.shape[1] == 2048:  # 13
                                    number = ssd
                                else:  # 103
                                    number = int(my_round(ssd * 0.7 + resnet * 0.3))
                            else:  # 84
                                number = int(my_round(ssd * 0.5 + resnet * 0.5))

    csv_writer.writerow([line[0], number])

print img_sum
print distribution_ssd
print distribution_resnet
