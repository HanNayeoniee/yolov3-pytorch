import torch
import numpy as np
import tqdm


def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse한다."""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path: str):
    """클래스 이름을 로드한다."""
    with open(path, "r") as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names


def init_weights_normal(m):
    """정규분포 형태로 가중치를 초기화한다."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes_original(prediction, rescaled_size: int, original_size: tuple):
    """Rescale bounding boxes to the original shape."""
    ow, oh = original_size
    resize_ratio = rescaled_size / max(original_size)

    # 적용된 패딩 계산
    if ow > oh:
        resized_w = rescaled_size
        resized_h = round(min(original_size) * resize_ratio)
        pad_x = 0
        pad_y = abs(resized_w - resized_h)
    else:
        resized_w = round(min(original_size) * resize_ratio)
        resized_h = rescaled_size
        pad_x = abs(resized_w - resized_h)
        pad_y = 0

    # Rescale bounding boxes
    prediction[:, 0] = (prediction[:, 0] - pad_x // 2) / resize_ratio
    prediction[:, 1] = (prediction[:, 1] - pad_y // 2) / resize_ratio
    prediction[:, 2] = (prediction[:, 2] - pad_x // 2) / resize_ratio
    prediction[:, 3] = (prediction[:, 3] - pad_y // 2) / resize_ratio

    # 예측 결과가 원본 이미지의 좌표를 넘어가지 못하게 한다.
    for i in range(prediction.shape[0]):
        for k in range(0, 3, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > ow:
                prediction[i][k] = ow

        for k in range(1, 4, 2):
            if prediction[i][k] < 0:
                prediction[i][k] = 0
            elif prediction[i][k] > oh:
                prediction[i][k] = oh

    return prediction

# 좌표의 형식을 x,y,w,h애서 x1,y1,x2,y2로 변환
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Compute AP", leave=False):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    for i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

# 박스2개 간의 iou 구하기
# 앵커박스의 중심좌표는 생각하지 않고 두 박스를 겹치게 놓은 후 겹치는 영역 구하기
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)  # 교집합 영역 구하기
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area  # 합집합 영역 구하기
    return inter_area / union_area

# 박스 2개의 iou를 구하기 위해서는 박스의 좌표를 x,y,w,h가 아닌 x1,y1,x2,y2로 알아야 함
# x1y1x2y2 변수는 좌표형식이 x1,y1,x2,y2형식을 가지는지 나타냄
def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
    # 박스의 좌표형식이 x1,y1,x2,y2가 아니라면 변환 수행
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

# 한 물체에 bbox가 여러개 쳐진 경우
# 예측한 박스들을 높은 score순으로 정렬한 후 가장 확률이 높은 bbox 1개만 남기기
def non_max_suppression(prediction, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # 임계값보다 작은 confidence score 필터링
        # image_pred의 shape=(10647, 85)
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        # 모든 이미지가 필터링되면 다음 이미지로 넘어가기
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        # max(1) : (values, indices)에서 max값 구하기
        # [0] : max class values 구하기
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # 최대 클래스 신뢰도와 클래스 레이블 얻기
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # NMS 수행
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    # (batch_size, pred_boxes_num, 7)
    # 7: x, y, w, h, conf(물체가 있는지에 대한 신뢰도), class_conf, class_pred
    # pred_boxes_num : 각 사진에 pred_boxes_num개의 상자가 있음
    return output

# YOLODetection 클래스에서 사용됨
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, device):
    # pred_boxes -> 예측한 bbox [batch_size, anchor개수, G, G, 85]
    # targets : ground truth [batch size, 6] -> (num, class, x, y, w, h)
    # num : 이미지 안에 있는 물체의 개수
    nB = pred_boxes.size(0)  # 배치 사이즈=1
    nA = pred_boxes.size(1)  # 앵커박스 개수=3
    nC = pred_cls.size(-1)  # 클래스 개수=80
    nG = pred_boxes.size(2)  # grid size= 13/ 26/ 52

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)  # 물체가 있는경우 1로 설정
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)  # 물체가 없는경우 1로 설정
    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # 예측한 클래스가 맞는지
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # Pr(object) * IOU
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device) # target x_ctr으로 ground truth값에 대한 것
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # target y_ctr
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # target width
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # target height
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)  # target class/ 정답 클래스

    # Convert to position relative to box
    # target shape: [index, class, x_ctr, y_ctr, w, h] (normalized)
    target_boxes = target[:, 2:6] * nG  # target은 1*1크기가 기준, 따라서 같은 grid로 비교하기 위해 grid만큼 곱함
    gxy = target_boxes[:, :2]  # target의 x, y
    gwh = target_boxes[:, 2:]  # target의 w, h

    # Get anchors with best iou
    # anchor box(예측)과 ground truth box(정답)간의 IOU 계산하기
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # (3, n) 크기, n은 하나의 이미지에 존재하는 ground truth 박스의 개수(물체의 개수)
    _, best_ious_idx = ious.max(0) # 각각의 ground truth박스에 가장 잘 맞는 anchor box 찾기, (1, n) 크기

    # Separate target values
    b, target_labels = target[:, :2].long().t()  # b는 batch-size index, 몇번째 배치인지(여기서는 1로 고정)
    gx, gy = gxy.t()  # ground truth의 x, y좌표
    gw, gh = gwh.t()  # ground truth의 w, h값
    gi, gj = gxy.long().t()  # 원래 x, y, w, h는 소수점까지 있는 값이기 때문에 정수값으로 만들기 위해 long()을 수행함

    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1  # 물체가 있을 때 1, 없으면 0
    noobj_mask[b, best_ious_idx, gj, gi] = 0  # 물체가 있을 때 0, 없으면 1

    # Set noobj mask to zero where iou exceeds ignore threshold
    # iou값이 임계치보다 크면 물체가 있을수도 있다고 판단해 0으로 설정
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    # target box 좌표 설정
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()  # 왼쪽상단 꼭지점을 기준으로 x좌표가 얼마나 움직였는지(변화량 구하기), offset
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()  # 왼쪽상단 꼭지점을 기준으로 y좌표가 얼마나 움직였는지, offset

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16) # exp()를 사용해 구했으므로 반대인 log()사용
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # One-hot encoding of label
    # target_label : 80개의 클래스 중 어느 클래스에 해당하는지
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()  # 예측을 잘 했는지
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)  # 정답, 예측간의 iou

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf