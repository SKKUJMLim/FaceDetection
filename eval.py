import numpy as np


from typing import List, Tuple

def calculate_iou(box1, box2):
    """
    두 박스의 IoU (Intersection over Union)를 계산합니다.
    :param box1: (x1, y1, x2, y2) 형태의 첫 번째 박스 좌표
    :param box2: (x1, y1, x2, y2) 형태의 두 번째 박스 좌표
    :return: IoU 값
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def calculate_true_positives(predictions: List[Tuple[int, int, int, int]], ground_truths: List[Tuple[int, int, int, int]], iou_threshold=0.5):
    """
    True Positives 계산 함수
    :param predictions: 모델이 예측한 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param ground_truths: 실제 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param iou_threshold: TP로 간주할 IoU 임계값
    :return: True Positives 수
    """

    if len(predictions) == 0:
        return 0  # 예측이 없을 경우 TP는 0

    true_positives = 0
    detected = [False] * len(ground_truths)

    for pred_box in predictions:
        for i, gt_box in enumerate(ground_truths):
            if not detected[i]:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    true_positives += 1
                    detected[i] = True
                    break

    return true_positives

def calculate_false_positives(predictions: List[Tuple[int, int, int, int]], ground_truths: List[Tuple[int, int, int, int]], iou_threshold=0.5):
    """
    False Positives 계산 함수
    :param predictions: 모델이 예측한 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param ground_truths: 실제 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param iou_threshold: TP로 간주할 IoU 임계값
    :return: False Positives 수
    """
    if len(predictions) == 0:
        return 0  # 예측이 없을 경우 FP는 0

    false_positives = 0
    for pred_box in predictions:
        match_found = False
        for gt_box in ground_truths:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                match_found = True
                break
        if not match_found:
            false_positives += 1
    return false_positives


def calculate_false_negatives(predictions: List[Tuple[int, int, int, int]], ground_truths: List[Tuple[int, int, int, int]], iou_threshold=0.5):
    """
    False Negatives 계산 함수
    :param predictions: 모델이 예측한 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param ground_truths: 실제 얼굴 위치 리스트 [(x1, y1, x2, y2), ...]
    :param iou_threshold: TP로 간주할 IoU 임계값
    :return: False Negatives 수
    """
    true_positives = calculate_true_positives(predictions, ground_truths, iou_threshold)
    false_negatives = len(ground_truths) - true_positives
    return false_negatives

def calculate_recall(true_positives, false_negatives):
    """
    Recall 계산 함수
    :param true_positives: 올바르게 감지된 얼굴 수 (True Positives)
    :param false_negatives: 감지되지 않은 얼굴 수 (False Negatives)
    :return: Recall 값
    """
    if (true_positives + false_negatives) == 0:
        return 0.0  # TP와 FN이 모두 0인 경우
    recall = true_positives / (true_positives + false_negatives)
    return recall


def calculate_precision(true_positives, false_positives):
    """
    Precision 계산 함수
    :param true_positives: 올바르게 감지된 얼굴 수 (True Positives)
    :param false_positives: 잘못 감지된 얼굴 수 (False Positives)
    :return: Precision 값
    """
    if (true_positives + false_positives) == 0:
        return 0.0  # TP와 FP가 모두 0인 경우
    precision = true_positives / (true_positives + false_positives)
    return precision


def calculate_overall_recall(images_results):
    """
    여러 이미지에 대한 Recall 계산 함수
    :param images_results: 각 이미지의 (true_positives, false_negatives) 값이 들어있는 리스트
    :return: 전체 Recall 값
    """
    total_true_positives = sum([result['true_positives'] for result in images_results])
    total_false_negatives = sum([result['false_negatives'] for result in images_results])

    overall_recall = calculate_recall(total_true_positives, total_false_negatives)
    return overall_recall


def calculate_overall_precision(images_results):
    """
    여러 이미지에 대한 Precision 계산 함수
    :param images_results: 각 이미지의 (true_positives, false_positives) 값이 들어있는 리스트
    :return: 전체 Precision 값
    """
    total_true_positives = sum([result['true_positives'] for result in images_results])
    total_false_positives = sum([result['false_positives'] for result in images_results])

    overall_precision = calculate_precision(total_true_positives, total_false_positives)
    return overall_precision


def calculate_precision_recall_at_threshold(predictions, ground_truths, iou_threshold):
    tp = calculate_true_positives(predictions, ground_truths, iou_threshold)
    fp = calculate_false_positives(predictions, ground_truths, iou_threshold)
    fn = len(ground_truths) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def calculate_average_precision(images, thresholds=np.linspace(0.1, 0.9, 9)):

    # thresholds == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    precisions = []
    recalls = []

    for threshold in thresholds:
        precisions_at_threshold = []
        recalls_at_threshold = []

        for image in images:
            predictions = image.get('predictions', [])
            ground_truths = image.get('ground_truths', [])
            precision, recall = calculate_precision_recall_at_threshold(predictions, ground_truths, threshold)
            precisions_at_threshold.append(precision)
            recalls_at_threshold.append(recall)

        # 각 threshold에서의 평균 Precision과 Recall
        mean_precision = np.mean(precisions_at_threshold)
        mean_recall = np.mean(recalls_at_threshold)

        precisions.append(mean_precision)
        recalls.append(mean_recall)

    # Precision-Recall 곡선 아래 면적 (AUC) 계산
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    sorted_indices = np.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]

    average_precision = np.trapz(sorted_precisions, sorted_recalls)  # AUC 계산

    return average_precision



if __name__ == '__main__':
    # 예시 데이터: 각 이미지의 예측 박스와 실제 박스
    test_images = [
        {
            'predictions': [(50, 50, 100, 100), (150, 150, 200, 200)],
            'ground_truths': [(150, 150, 200, 200), (55, 55, 105, 105)]
        },
        {
            'predictions': [(30, 30, 80, 80), (170, 170, 220, 220)],
            'ground_truths': [(25, 25, 75, 75), (160, 160, 210, 210), (100, 100, 150, 150)]
        },
        # 더 많은 이미지 데이터 추가 가능
        # {'predictions': [(312, 612, 340, 657)], 'ground_truths': [(313, 613, 341, 661)]}

    ]

    images_results = []
    iou_threshold = 0.5

    for image in test_images:
        tp = calculate_true_positives(image['predictions'], image['ground_truths'], iou_threshold)
        fn = calculate_false_negatives(image['predictions'], image['ground_truths'], iou_threshold)
        fp = calculate_false_positives(image['predictions'], image['ground_truths'], iou_threshold)
        images_results.append({'true_positives': tp, 'false_positives': fp, 'false_negatives': fn})

    overall_recall = calculate_overall_recall(images_results)
    overall_precision = calculate_overall_precision(images_results)

    average_precision = calculate_average_precision(test_images)

    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Average Precision: {average_precision:.2f}")