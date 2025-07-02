# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from mmpose.evaluation.functional.nms import oks_iou


def _compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def _track_by_iou(res, results_last, thr):
    """Get track id using IoU tracking greedily."""

    """ # 原程式碼代碼
    bbox = list(np.squeeze(res.pred_instances.bboxes, axis=0))

    max_iou_score = -1
    max_index = -1
    match_result = {}
    for index, res_last in enumerate(results_last):
        bbox_last = list(np.squeeze(res_last.pred_instances.bboxes, axis=0))

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index].track_id
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result
    """

    if isinstance(res.pred_instances.bboxes, torch.Tensor):
        # 這邊就是如果發現現在的 res..... 這個是張量的話，我們需要先繞過cpu再把它變成列表
        bboxes = res.pred_instances.bboxes.cpu().numpy()  # bboxes = [邊界框(x1, x2, y1, y2), 可信度分數]
    else:
        bboxes = res.pred_instances.bboxes

    track_ids = [] # 自訂
    match_results = [] # 自訂

    if bboxes.shape[0] == 0: # 檢查目前的bboxes有沒有邊界框，
        return [], results_last, [] # 因為這幀沒有偵測到邊界框，所以只有回傳上一幀的追蹤結果。[ track_id, results_last, match_result ]

    for i in range(bboxes.shape[0]):
        bbox = list(bboxes[i]) # 準備一個一個去對每個邊界框
        max_iou_score = -1
        max_index = -1
        match_result = {}

        # 這邊都一樣抄原本的代碼
        for index, res_last in  enumerate(results_last): # 準備將此幀的此張臉去和過去的臉部 IOU 做比對。
            if isinstance(res_last.pred_instances.bboxes, torch.Tensor):
                last_bboxes = res_last.pred_instances.bboxes.cpu().numpy()
            else:
                last_bboxes = res_last.pred_instances.bboxes
            
            if last_bboxes.shape[0] == 0:
                print(f"有一幀找不到過去的邊界框，編號 {index}")
                continue

            bbox_last = list(last_bboxes[0])
            iou_score = _compute_iou(bbox, bbox_last)

            # 這邊就是跟我們影像作業的 max_iou 很像，用現在此幀每張臉去輪流比對過去(results_last)的所有臉，
            # 去取最大的 max_iou_score，還有更新最佳配對的 index
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index 
            
        if max_iou_score > thr:
            track_id = results_last[max_index].track_id 
            # results_last[0]的 ID是零(其實我也不是很清楚results_last到底塞了哪些東西。 反正就按照原作者的寫法)
            match_result = results_last[max_index]
            del results_last[max_index]
        else:
            track_id = -1

        track_ids.append(track_id)
        match_results.append(match_result)

    return track_ids, results_last, match_results





def _track_by_oks(res, results_last, thr, sigmas=None):
    """原代碼
    # Get track id using OKS tracking greedily.
    keypoint = np.concatenate((res.pred_instances.keypoints,
                               res.pred_instances.keypoint_scores[:, :, None]),
                              axis=2)
    keypoint = np.squeeze(keypoint, axis=0).reshape((-1))
    area = np.squeeze(res.pred_instances.areas, axis=0)
    max_index = -1
    match_result = {}

    if len(results_last) == 0:
        return -1, results_last, match_result

    keypoints_last = np.array([
        np.squeeze(
            np.concatenate(
                (res_last.pred_instances.keypoints,
                 res_last.pred_instances.keypoint_scores[:, :, None]),
                axis=2),
            axis=0).reshape((-1)) for res_last in results_last
    ])
    area_last = np.array([
        np.squeeze(res_last.pred_instances.areas, axis=0)
        for res_last in results_last
    ])

    oks_score = oks_iou(
        keypoint, keypoints_last, area, area_last, sigmas=sigmas)

    max_index = np.argmax(oks_score)

    if oks_score[max_index] > thr:
        track_id = results_last[max_index].track_id
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1
    """
           
    # 改
    # 從傳入的 res 物件中獲取所有人的關鍵點、分數和面積
    if isinstance(res.pred_instances.keypoints, torch.Tensor):
        keypoints_all = res.pred_instances.keypoints.cpu().numpy()
        scores_all = res.pred_instances.keypoint_scores.cpu().numpy()
        areas_all = res.pred_instances.areas.cpu().numpy()
    else:
        keypoints_all = res.pred_instances.keypoints
        scores_all = res.pred_instances.keypoint_scores
        areas_all = res.pred_instances.areas

    track_ids = []
    match_results = []

    # 檢查當前幀是否有偵測到任何人
    if keypoints_all.shape[0] == 0:
        return [], results_last, []

    # 遍歷當前幀的每一個人
    for i in range(keypoints_all.shape[0]):
        # 提取當前這個人的關鍵點資訊
        keypoint_i_with_score = np.concatenate(
            (keypoints_all[i], scores_all[i][..., None]), axis=1)
        keypoint_i = keypoint_i_with_score.ravel() # 攤平成一維陣列
        area_i = areas_all[i]

        max_oks_score = -1
        max_index = -1
        match_result = {}
        
        # 如果上一幀沒有任何紀錄，直接判定為新人
        if len(results_last) == 0:
            track_id = -1
            track_ids.append(track_id)
            match_results.append(match_result)
            continue

        # 準備上一幀的所有人數據，以便進行 OKS 計算
        keypoints_last = []
        area_last = []
        for res_last in results_last:
            if isinstance(res_last.pred_instances.keypoints, torch.Tensor):
                keypoints_last_person = res_last.pred_instances.keypoints.cpu().numpy()[0]
                scores_last_person = res_last.pred_instances.keypoint_scores.cpu().numpy()[0]
                area_last_person = res_last.pred_instances.areas.cpu().numpy()[0]
            else:
                keypoints_last_person = res_last.pred_instances.keypoints[0]
                scores_last_person = res_last.pred_instances.keypoint_scores[0]
                area_last_person = res_last.pred_instances.areas[0]

            keypoint_last_with_score = np.concatenate(
                (keypoints_last_person, scores_last_person[..., None]), axis=1)
            keypoints_last.append(keypoint_last_with_score.ravel())
            area_last.append(area_last_person)

        keypoints_last = np.array(keypoints_last)
        area_last = np.array(area_last)

        # 計算當前這個人與上一幀所有人的 OKS 分數
        oks_scores = oks_iou(keypoint_i, keypoints_last, area_i, area_last, sigmas)
        
        # 找到分數最高的那個人
        max_index = np.argmax(oks_scores)
        max_oks_score = oks_scores[max_index]

        # 根據閾值判斷是否為同一個人
        if max_oks_score > thr:
            track_id = results_last[max_index].track_id
            match_result = results_last[max_index]
            del results_last[max_index]
        else:
            track_id = -1
        
        track_ids.append(track_id)
        match_results.append(match_result)

    return track_ids, results_last, match_results
