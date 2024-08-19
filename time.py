from scrfd_detector import SCRFD, distance2bbox, distance2kps
import datetime
import cv2

def check_time(img_path, detector):

    img = cv2.imread(img_path)

    ta = datetime.datetime.now()
    bboxes, kpss = detector.detect(img, 0.5, input_size=(640, 640))
    # bboxes, kpss = detector.detect(img, 0.5)
    tb = datetime.datetime.now()
    print('all cost:', (tb-ta).total_seconds()*1000)



if __name__ == '__main__':

    # detector = SCRFD(model_file='./onnx/scrfd_500m.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_1g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_2.5g.onnx')
    # detector = SCRFD(model_file='./onnx/scrfd_10g.onnx')
    detector = SCRFD(model_file='./onnx/scrfd_34g.onnx')
    detector.prepare(-1)

    check_time('data/jpg/004556.jpg', detector)