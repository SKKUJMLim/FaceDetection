# !pip install onnxruntime
# !pip install matplotlib
# !pip install opencv-python

import datetime
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from pathlib import Path

from scrfd_detector import SCRFD, distance2bbox, distance2kps
from eval import calculate_true_positives, calculate_false_negatives, \
    calculate_false_positives, calculate_overall_recall, calculate_overall_precision, calculate_average_precision


def xml_parsing(img_file_path, xml_file_path, detector, output_path):
    xml_paths = os.listdir(xml_file_path)
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)

    test_images = []
    not_detection = []

    for xml_file in xml_paths:

        print("detection... ", xml_file)

        tree = ET.parse(xml_file_path + xml_file)
        root = tree.getroot()

        img_path, _ = os.path.splitext(xml_file)
        img = cv2.imread(img_file_path + img_path + ".jpg")
        # opencv의 rectangle()는 인자로 들어온 이미지 배열에 그대로 사각형을 그려주므로 별도의 이미지 배열에 그림 작업 수행.
        draw_img = img.copy()

        ta = datetime.datetime.now()
        bboxes, kpss = detector.detect(img, 0.5, input_size=(640, 640))
        # bboxes, kpss = detector.detect(img, 0.5)
        tb = datetime.datetime.now()
        # print('all cost:', (tb-ta).total_seconds()*1000)

        predictions = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            xmin, ymin, xmax, ymax, score = bbox.astype(np.int32)

            prediction= (xmin, ymin, xmax, ymax)
            predictions.append(prediction)

            # prediction은 빨간색으로 box 표시
            cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color=red_color, thickness=7)

        ground_truths = []

        for object in root.iter('object'):
            index = object.find('index').text

            width = (object.find('bndbox').findtext('width'))
            height = int(object.find('bndbox').findtext('height'))

            xmin = int(object.find('bndbox').findtext('xmin'))
            ymin = int(object.find('bndbox').findtext('ymin'))

            xmax = int(object.find('bndbox').findtext('xmax'))
            ymax = int(object.find('bndbox').findtext('ymax'))

            ground_truth = (xmin, ymin, xmax, ymax)
            ground_truths.append(ground_truth)

            # Grount truth는 녹색으로 box 표시
            cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color=green_color, thickness=5)

            # draw_img 배열의 좌상단 좌표에 Size 표시
            cv2.putText(draw_img, str(width) + "x" + str(height), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                        red_color, thickness=2)

        # img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + img_path + ".jpg", draw_img)

        if len(predictions) != len(ground_truths):
            not_detection.append(img_path)

        test_images.append({"predictions": predictions, "ground_truths": ground_truths})

        print("Not Detecitonn List")
        print(not_detection)

    return test_images


if __name__ == '__main__':

    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA GPU를 사용할 수 없습니다.")
    else:
        print("CUDA GPU를 사용할 수 있습니다.")

    img_file_path = 'data/jpg/'
    xml_file_path = 'data/xml/'
    output_path = 'data/ground_truth/'

    # detector = SCRFD(model_file='./onnx/scrfd_500m.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_1g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_2.5g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_10g.onnx')
    detector = SCRFD(model_file='./onnx/scrfd_34g.onnx')
    detector.prepare(-1)

    folder_path = Path(output_path)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"폴더를 생성했습니다: {folder_path}")
    else:
        print(f"폴더가 이미 존재합니다: {folder_path}")

    test_images = xml_parsing(img_file_path, xml_file_path, detector, output_path)

    # print(len(test_images))
    # print(test_images)

    images_results = []
    iou_threshold = 0.5

    for image in test_images:
        predictions = image.get('predictions', [])
        ground_truths = image.get('ground_truths', [])

        tp = calculate_true_positives(predictions, ground_truths, iou_threshold)
        fn = calculate_false_negatives(predictions, ground_truths, iou_threshold)
        fp = calculate_false_positives(predictions, ground_truths, iou_threshold)
        images_results.append({'true_positives': tp, 'false_positives': fp, 'false_negatives': fn})

    # print("images_results == ", images_results)
    overall_recall = calculate_overall_recall(images_results)
    overall_precision = calculate_overall_precision(images_results)

    average_precision = calculate_average_precision(test_images)

    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Average Precision: {average_precision:.2f}")
