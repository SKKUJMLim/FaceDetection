# !pip install onnxruntime
# !pip install matplotlib
# !pip install opencv-python

import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from pathlib import Path

from scrfd_detector import SCRFD,distance2bbox,distance2kps


def detection(img_file_path, xml_file_path, output_path, detector):


    img_paths = os.listdir(img_file_path)
    
    for img_file in img_paths:
    
        img = cv2.imread(img_file_path + img_file)
        # opencv의 rectangle()는 인자로 들어온 이미지 배열에 그대로 사각형을 그려주므로 별도의 이미지 배열에 그림 작업 수행.
        draw_img = img.copy()
        
        faceArea_list = list()

        # for _ in range(1):
        #     print(_)
        ta = datetime.datetime.now()
        # bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640))
        bboxes, kpss = detector.detect(img, 0.5)
        tb = datetime.datetime.now()
        #print('all cost:', (tb-ta).total_seconds()*1000)


        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            xmin,ymin,xmax,ymax,score = bbox.astype(np.int32)

            faceArea = img[ymin:ymax, xmin:xmax].copy()
            faceArea_list.append(faceArea)

            green_color = (0, 255, 0)
            red_color = (0, 0, 255)
            cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color=green_color, thickness=5)

        # img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + img_file, draw_img)

if __name__ == '__main__':

    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA GPU를 사용할 수 없습니다.")
    else:
        print("CUDA GPU를 사용할 수 있습니다.")
        
        # 8개의 모델 중, 필요에 따라 하나의 모델 선정

    # detector = SCRFD(model_file='./onnx/scrfd_500m.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_1g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_2.5g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_10g.onnx')
    detector = SCRFD(model_file='./onnx/scrfd_34g.onnx')
    detector.prepare(-1)

    img_file_path = 'data/jpg/'
    xml_file_path = 'data/xml/'
    output_path = 'data/prediction/'

    folder_path = Path(output_path)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"폴더를 생성했습니다: {folder_path}")
    else:
        print(f"폴더가 이미 존재합니다: {folder_path}")

    detection(img_file_path, xml_file_path, output_path, detector)