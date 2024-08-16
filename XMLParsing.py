# !pip install onnxruntime
# !pip install matplotlib
# !pip install opencv-python

import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def xml_parsing(img_file_path, xml_file_path, output_path):
    
    xml_paths = os.listdir(xml_file_path)
    
    for xml_file in xml_paths:
     
        print(xml_file)
    
        tree = ET.parse(xml_file_path + xml_file)
        root = tree.getroot()
        
        img_path, _ = os.path.splitext(xml_file)
        img = cv2.imread(img_file_path + img_path + ".jpg")
        # opencv의 rectangle()는 인자로 들어온 이미지 배열에 그대로 사각형을 그려주므로 별도의 이미지 배열에 그림 작업 수행. 
        draw_img = img.copy()
        
        
        for object in root.iter('object'):

          index = object.find('index').text

          width = (object.find('bndbox').findtext('width'))
          height = int(object.find('bndbox').findtext('height'))

          xmin = int(object.find('bndbox').findtext('xmin'))
          ymin = int(object.find('bndbox').findtext('ymin'))

          xmax = int(object.find('bndbox').findtext('xmax'))
          ymax = int(object.find('bndbox').findtext('ymax'))
          
          # draw_img 배열의 좌상단 우하단 좌표에 녹색으로 box 표시 
          green_color=(0, 255, 0)
          red_color=(0, 0, 255)
          cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color=green_color, thickness=5)
          
          # draw_img 배열의 좌상단 좌표에 Size 표시
          cv2.putText(draw_img, str(width) + "x" + str(height), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.2, red_color, thickness=2)
        
        # img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + img_path + ".jpg", draw_img)



if __name__ == '__main__':

    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("CUDA GPU를 사용할 수 없습니다.")
    else:
        print("CUDA GPU를 사용할 수 있습니다.")
        
    img_file_path = 'data/jpg/'
    xml_file_path = 'data/xml/'
    output_path = 'data/ground_truth/'

    folder_path = Path(output_path)
    
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        print(f"폴더를 생성했습니다: {folder_path}")
    else:
        print(f"폴더가 이미 존재합니다: {folder_path}")
    
    xml_parsing(img_file_path, xml_file_path, output_path)