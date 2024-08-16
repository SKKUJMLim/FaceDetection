import numpy as np

def iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def average_precision(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]

    num_gts = len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    detected = []

    for i, pred_box in enumerate(pred_boxes):
        ious = np.array([iou(pred_box, gt_box) for gt_box in gt_boxes])
        max_iou_index = np.argmax(ious)
        max_iou = ious[max_iou_index]

        if max_iou >= iou_threshold and max_iou_index not in detected:
            tp[i] = 1
            detected.append(max_iou_index)
        else:
            fp[i] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / float(num_gts)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, np.finfo(np.float64).eps)

    ap = average_precision(recalls, precisions)
    return ap

if __name__ == '__main__':
    # 예시로 사용할 예측값과 실제값
    pred_boxes = np.array([[50, 50, 150, 150], [30, 30, 100, 100], [200, 200, 300, 300]])  # 예측 박스 3개
    pred_scores = np.array([0.9, 0.9, 0.8])  # 각각의 예측 박스에 대한 신뢰도 점수
    gt_boxes = np.array([[40, 40, 160, 160], [205, 205, 305, 305]])  # 실제 박스 2개

    # AP 계산
    ap = compute_ap(pred_boxes, pred_scores, gt_boxes)
    print(f'AP: {ap}')