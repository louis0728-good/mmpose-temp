# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer.infer import ModelType
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmpose.apis import (_track_by_iou, _track_by_oks, collate_pose_sequence,
                         convert_keypoint_definition, extract_pose_sequence)
from mmpose.registry import INFERENCERS
from mmpose.structures import PoseDataSample, merge_data_samples
from .base_mmpose_inferencer import BaseMMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='pose-estimation-3d')
@INFERENCERS.register_module()
class Pose3DInferencer(BaseMMPoseInferencer):
    """The inferencer for 3D pose estimation.

    Args:
        model (str, optional): Pretrained 2D pose estimation algorithm.
            It's the path to the config file or the model name defined in
            metafile. For example, it could be:

            - model alias, e.g. ``'body'``,
            - config name, e.g. ``'simcc_res50_8xb64-210e_coco-256x192'``,
            - config path

            Defaults to ``None``.
        weights (str, optional): Path to the checkpoint. If it is not
            specified and "model" is a model name of metafile, the weights
            will be loaded from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the
            available device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmpose".
        det_model (str, optional): Config path or alias of detection model.
            Defaults to None.
        det_weights (str, optional): Path to the checkpoints of detection
            model. Defaults to None.
        det_cat_ids (int or list[int], optional): Category id for
            detection model. Defaults to None.
        output_heatmaps (bool, optional): Flag to visualize predicted
            heatmaps. If set to None, the default setting from the model
            config will be used. Default is None.
    """

    preprocess_kwargs: set = {
        'bbox_thr', 'nms_thr', 'bboxes', 'use_oks_tracking', 'tracking_thr',
        'disable_norm_pose_2d'
    }
    forward_kwargs: set = {'disable_rebase_keypoint'}
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_bbox',
        'radius',
        'thickness',
        'num_instances',
        'kpt_thr',
        'vis_out_dir',
    }
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 pose2d_model: Optional[Union[ModelType, str]] = None,
                 pose2d_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmpose',
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, Tuple]] = None,
                 show_progress: bool = False) -> None:

        init_default_scope(scope)
        super().__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            show_progress=show_progress)
        self.model = revert_sync_batchnorm(self.model)

        # assign dataset metainfo to self.visualizer
        self.visualizer.set_dataset_meta(self.model.dataset_meta)

        # initialize 2d pose estimator
        self.pose2d_model = Pose2DInferencer(
            pose2d_model if pose2d_model else 'human', pose2d_weights, device,
            scope, det_model, det_weights, det_cat_ids)

        # helper functions
        self._keypoint_converter = partial(
            convert_keypoint_definition,
            pose_det_dataset=self.pose2d_model.model.
            dataset_meta['dataset_name'],
            pose_lift_dataset=self.model.dataset_meta['dataset_name'],
        )

        self._pose_seq_extractor = partial(
            extract_pose_sequence,
            causal=self.cfg.test_dataloader.dataset.get('causal', False),
            seq_len=self.cfg.test_dataloader.dataset.get('seq_len', 1),
            step=self.cfg.test_dataloader.dataset.get('seq_step', 1))

        self._video_input = False
        self._buffer = defaultdict(list)

    def preprocess_single(self,
                          input: InputType,
                          index: int,
                          bbox_thr: float = 0.3,
                          nms_thr: float = 0.3,
                          bboxes: Union[List[List], List[np.ndarray],
                                        np.ndarray] = [],
                          use_oks_tracking: bool = False,
                          tracking_thr: float = 0.3,
                          disable_norm_pose_2d: bool = False):
        """Process a single input into a model-feedable format.

        Args:
            input (InputType): The input provided by the user.
            index (int): The index of the input.
            bbox_thr (float, optional): The threshold for bounding box
                detection. Defaults to 0.3.
            nms_thr (float, optional): The Intersection over Union (IoU)
                threshold for bounding box Non-Maximum Suppression (NMS).
                Defaults to 0.3.
            bboxes (Union[List[List], List[np.ndarray], np.ndarray]):
                The bounding boxes to use. Defaults to [].
            use_oks_tracking (bool, optional): A flag that indicates
                whether OKS-based tracking should be used. Defaults to False.
            tracking_thr (float, optional): The threshold for tracking.
                Defaults to 0.3.
            disable_norm_pose_2d (bool, optional): A flag that indicates
                whether 2D pose normalization should be used.
                Defaults to False.

        Yields:
            Any: The data processed by the pipeline and collate_fn.

        This method first calculates 2D keypoints using the provided
        pose2d_model. The method also performs instance matching, which
        can use either OKS-based tracking or IOU-based tracking.
        """

        # calculate 2d keypoints
        # 執行2D姿態偵測
        results_pose2d = next(
            self.pose2d_model(
                input,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                bboxes=bboxes,
                merge_results=False,
                return_datasamples=True))['predictions']

        for ds in results_pose2d:
            ds.pred_instances.set_field(
                (ds.pred_instances.bboxes[..., 2:] -
                 ds.pred_instances.bboxes[..., :2]).prod(-1), 'areas')

        if not self._video_input:
            height, width = results_pose2d[0].metainfo['ori_shape']

            # Clear the buffer if inputs are individual images to prevent
            # carryover effects from previous images
            self._buffer.clear()

        else:
            height = self.video_info['height']
            width = self.video_info['width']
        #img_path = results_pose2d[0].metainfo['img_path'] # 原

        # 改
        if not results_pose2d:
            img_path = ''
        else:
            img_path = results_pose2d[0].metainfo['img_path']

        # 將單人 result 列表合併成一個多人的 data 樣本
        # merged_results = merge_data_samples(results_pose2d) # 改的
        # instance matching
        # 回歸最穩定的「單獨追蹤」模式
        if use_oks_tracking:
            _track = partial(_track_by_oks)
        else:
            _track = _track_by_iou

        # 這邊一直提到的 buffer 就是「專門儲存不同幀之間需要保留的資訊」
        # self._buffer['results_pose2d_last']: 就是上一幀的2D偵測結果
        # self._buffer["pose_est_results_list"]: 累積所有幀的2D pose結果，格式會長的像 [[frame1人], [frame2人], ...]
        # self._buffer["pose2d_results"]: 把這一幀的 pose2d 結果丟進去
        # self._buffer["next_id"]: 下一個要分配的ID，這樣每個人都可以有唯一的編號

        for result in results_pose2d: # 這裡改回來了
            track_id, self._buffer['results_pose2d_last'], _ = _track(
                result, self._buffer['results_pose2d_last'], tracking_thr)
        
        # 我要用迴圈遍歷每個人，把追蹤到的 track_id 分配回去
        # for result, track_id in zip(results_pose2d, track_ids): # 改的
            # 如果 -1 代表是新人
            if track_id == -1:
                # 先檢查這個新人的關鍵點數量是否夠多，以免是誤抓
                # pred_instances = result.pred_instances.cpu().numpy()
                # keypoints = pred_instances.keypoints
                # if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    # 從 buffer 中獲取下一個可用的新 ID
                next_id = self._buffer.get('next_id', 0)
                result.set_field(next_id, 'track_id')
                # 更新下一輪的 ID
                self._buffer['next_id'] = next_id + 1

                """ 先刪掉?
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    # 上面講英文，其實就是: 如果關鍵點太少，視為無效偵測，將 track_id 設為 -1
                    result.pred_instances.keypoints[..., 1] = -10
                    result.pred_instances.bboxes *= 0
                    result.set_field(-1, 'track_id')
                """

            else:
                result.set_field(track_id, 'track_id')
        
        # 更新 buffer，確保視覺化時使用的是帶有 track_id 的最新結果
        if results_pose2d:
            self._buffer['pose2d_results'] = merge_data_samples(results_pose2d)

        # convert keypoints
        # 轉換與緩存
        results_pose2d_converted = [ds.cpu().numpy() for ds in results_pose2d]
        for ds in results_pose2d_converted:
            ds.pred_instances.keypoints = self._keypoint_converter(
                ds.pred_instances.keypoints)
        self._buffer['pose_est_results_list'].append(results_pose2d_converted)

        # extract and pad input pose2d sequence
        # 提取序列
        pose_results_2d = self._pose_seq_extractor(
            self._buffer['pose_est_results_list'],
            frame_idx=index if self._video_input else 0)
        
        if not pose_results_2d:# 改
            return []
        
        # 複製與正規化
        causal = self.cfg.test_dataloader.dataset.get('causal', False)
        target_idx = -1 if causal else len(pose_results_2d) // 2
        stats_info = self.model.dataset_meta.get('stats_info', {})
        bbox_center = stats_info.get('bbox_center', None)
        bbox_scale = stats_info.get('bbox_scale', None)

        pose_results_2d_copy = []
        for pose_res in pose_results_2d:
            pose_res_copy = []
            for data_sample in pose_res:

                kpts = data_sample.pred_instances.keypoints
                if kpts is None or kpts.size == 0:
                    print("[DEBUG] 此人沒有有效keypoints，跳過這個人")
                    continue

                data_sample_copy = PoseDataSample()
                data_sample_copy.gt_instances = \
                    data_sample.gt_instances.clone()
                data_sample_copy.pred_instances = \
                    data_sample.pred_instances.clone()
                # data_sample_copy.track_id = data_sample.track_id # 原本

                # 穩健地獲取和設定track_id
                track_id = getattr(data_sample, 'track_id', -1) # 改
                data_sample_copy.set_field(track_id, 'track_id') # 改

                kpts = data_sample.pred_instances.keypoints
                bboxes = data_sample.pred_instances.bboxes
                keypoints = []
                for k in range(len(kpts)):
                    kpt = kpts[k]
                    if not disable_norm_pose_2d:
                        bbox = bboxes[k]
                        center = np.array([[(bbox[0] + bbox[2]) / 2,
                                            (bbox[1] + bbox[3]) / 2]])
                        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                        keypoints.append((kpt[:, :2] - center) / scale *
                                         bbox_scale + bbox_center)
                    else:
                        keypoints.append(kpt[:, :2])

                print("[DEBUG] preprocess_single keypoints 的 shape:", np.array(keypoints).shape)
                data_sample_copy.pred_instances.set_field(
                    np.array(keypoints), 'keypoints')
                pose_res_copy.append(data_sample_copy)

            pose_results_2d_copy.append(pose_res_copy)
        # 整理序列
        pose_sequences_2d = collate_pose_sequence(pose_results_2d_copy, True,
                                                  target_idx)
        if not pose_sequences_2d or all(len(p)==0 for p in pose_results_2d_copy):
            print("[DEBUG] 整幀沒有任何有效的人，跳過這幀")
            return []
        
        print('[Debug], 目前pose_sequences_2d長度是:', len(pose_sequences_2d), '代表有此幀有這麼多人\n') # 改

        data_list = []
        for i, pose_seq in enumerate(pose_sequences_2d):
            print(f"  [Debug] 第 {i} 個人的 keypoints shape: {pose_seq.pred_instances.keypoints.shape}\n") # 改
            # 每個人都有一個獨立序列
            data_info = dict()

            keypoints_2d = pose_seq.pred_instances.keypoints
            keypoints_2d = np.squeeze(
                keypoints_2d,
                axis=1) if keypoints_2d.ndim == 4 else keypoints_2d

            T, K, C = keypoints_2d.shape
            # 時間幀數, 關鍵點數量, 座標
            data_info['keypoints'] = keypoints_2d
            # keypoints_visible 和 factor 的長度都是 T
            data_info['keypoints_visible'] = np.ones((
                T,
                K,
            ),
                                                     dtype=np.float32)
            data_info['lifting_target'] = np.zeros((T, K, 3), dtype=np.float32) # 改
            # 這裡要把範圍改成 T, K, 3 不然會報錯說
            # non-broadcastable output operand with shape (1,17,3) doesn't match the broadcast shape (7,17,3)
            # 簡單講就是他規定只能 一人， 但是我這裡卻實際輸入 很多人。
            data_info['factor'] = np.zeros((T, ), dtype=np.float32)
            data_info['lifting_target_visible'] = np.ones((T, K, 1),
                                                          dtype=np.float32)
            data_info['camera_param'] = dict(w=width, h=height)

            data_info.update(self.model.dataset_meta)
            data_info = self.pipeline(data_info)
            data_info['data_samples'].set_field(
                img_path, 'img_path', field_type='metainfo')
            data_list.append(data_info)

        return data_list

    @torch.no_grad()
    def forward(self,
                inputs: Union[dict, tuple],
                disable_rebase_keypoint: bool = False):
        """Perform forward pass through the model and process the results.

        Args:
            inputs (Union[dict, tuple]): The inputs for the model.
            disable_rebase_keypoint (bool, optional): Flag to disable rebasing
                the height of the keypoints. Defaults to False.

        Returns:
            list: A list of data samples, each containing the model's output
                results.
        """
        print(f"[DEBUG] 3D模型運行裝置: {next(self.model.parameters()).device}")
        pose_lift_results = self.model.test_step(inputs)
        

        # Post-processing of pose estimation results
        pose_est_results_converted = self._buffer['pose_est_results_list'][-1]
        for idx, pose_lift_res in enumerate(pose_lift_results):
            scores = pose_lift_res.pred_instances.keypoint_scores
            if np.mean(scores) < 0.3:
                print(f"有一幀畫面可信度分數太低，跳過 {idx}")

            # Update track_id from the pose estimation results
            pose_lift_res.track_id = pose_est_results_converted[idx].get(
                'track_id', 1e4)

            # align the shape of output keypoints coordinates and scores
            # 移除對模型輸出多餘的squeeze操作
            keypoints = pose_lift_res.pred_instances.keypoints
            # keypoint_scores = pose_lift_res.pred_instances.keypoint_scores # 原本就有，但先註解
            """ # 改
            if keypoint_scores.ndim == 3:
                print("[DEBUG] keypoint_scores 的大小:", keypoint_scores.shape)
                pose_lift_results[idx].pred_instances.keypoint_scores = \
                    np.squeeze(keypoint_scores, axis=1) 
            
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1) """

            # Invert x and z values of the keypoints
            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 0] = -keypoints[..., 0]
            keypoints[..., 2] = -keypoints[..., 2]

            # If rebase_keypoint_height is True, adjust z-axis values
            if not disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            # pose_lift_results[idx].pred_instances.keypoints = keypoints # 原
            pose_lift_res.pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        """
        data_samples = [merge_data_samples(pose_lift_results)]

        # 改 這邊做視覺化修正，因為視覺化工具一次只能顯示一幀，我們選擇序列的中心幀作為代表
        center_idx = data_samples[0].pred_instances.keypoints.shape[1] // 2

        # 從 keypoints 中只選取中心幀的資料
        center_kpts = data_samples[0].pred_instances.keypoints[:, center_idx, :, :]
        data_samples[0].pred_instances.keypoints = center_kpts

        # 同樣，從 keypoint_scores 中也只選取中心幀的資料
        if 'keypoint_scores' in data_samples[0].pred_instances:
            center_scores = data_samples[0].pred_instances.keypoint_scores[:, center_idx, :]
            data_samples[0].pred_instances.keypoint_scores = center_scores
        """

        """
        # 先對每個人單獨提取中心幀，後來這裡叫我刪掉。
        for data_sample in pose_lift_results:  #改
            kpts = data_sample.pred_instances.keypoints
            if kpts.ndim == 4:
                center_idx = kpts.shape[1] // 2
                kpts_center = kpts[:, center_idx:center_idx+1, :, :]
                data_sample.pred_instances.keypoints = kpts[:, center_idx, :, :]

            if 'keypoint_scores' in data_sample.pred_instances:
                scores = data_sample.pred_instances.keypoint_scores
                if scores.ndim == 3:
                    center_idx = scores.shape[1] // 2
                    data_sample.pred_instances.keypoint_scores = scores[:, center_idx, :]
        """

        # 最後再合併成一個DataSample以供視覺化
        # data_samples = [merge_data_samples(pose_lift_results)] # 將物件放入一個列表中 # 後來改成不要用?


        print("[DEBUG] pose_lift_results數量(總共幾個人):", len(pose_lift_results))
        for i, d in enumerate(pose_lift_results):
            print(f"  第{i}個人的 keypoints shape:", d.pred_instances.keypoints.shape)

        # return data_samples # 這個隨著merge不要用，也被取消了。
        return pose_lift_results

    
    def visualize(self,
                  inputs: list,
                  preds: List[PoseDataSample],
                  return_vis: bool = False,
                  show: bool = False,
                  draw_bbox: bool = False,
                  wait_time: float = 0,
                  radius: int = 3,
                  thickness: int = 1,
                  kpt_thr: float = 0.3,
                  num_instances: int = 1,
                  vis_out_dir: str = '',
                  window_name: str = '',
                  window_close_event_handler: Optional[Callable] = None
                  ) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            return_vis (bool): Whether to return images with predicted results.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (ms). Defaults to 0
            draw_bbox (bool): Whether to draw the bounding boxes.
                Defaults to False
            radius (int): Keypoint radius for visualization. Defaults to 3
            thickness (int): Link thickness for visualization. Defaults to 1
            kpt_thr (float): The threshold to visualize the keypoints.
                Defaults to 0.3
            vis_out_dir (str, optional): Directory to save visualization
                results w/o predictions. If left as empty, no file will
                be saved. Defaults to ''.
            window_name (str, optional): Title of display window.
            window_close_event_handler (callable, optional):

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if (not return_vis) and (not show) and (not vis_out_dir):
            return

        if getattr(self, 'visualizer', None) is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.radius = radius
        self.visualizer.line_width = thickness
        det_kpt_color = self.pose2d_model.visualizer.kpt_color
        det_dataset_skeleton = self.pose2d_model.visualizer.skeleton
        det_dataset_link_color = self.pose2d_model.visualizer.link_color
        self.visualizer.det_kpt_color = det_kpt_color
        self.visualizer.det_dataset_skeleton = det_dataset_skeleton
        self.visualizer.det_dataset_link_color = det_dataset_link_color

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img = mmcv.imread(single_input, channel_order='rgb')
            elif isinstance(single_input, np.ndarray):
                img = mmcv.bgr2rgb(single_input)
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            # 對每個人取中心幀
            kpts = pred.pred_instances.keypoints
            if kpts.ndim ==4:
                center_idx = kpts.shape[1]//2
                # kpts_center = kpts[:, center_idx:center_idx+1, :, :] # 原
                kpts_center = kpts[:, center_idx, :, :] # 改
                pred.pred_instances.keypoints = kpts_center

            if hasattr(pred.pred_instances, "keypoint_scores"):
                scores = pred.pred_instances.keypoint_scores
                if scores is not None and scores.ndim ==3:
                    center_idx = scores.shape[1]//2
                    # scores_center = scores[:, center_idx:center_idx+1, :] # 原
                    scores_center = scores[:, center_idx, :]
                    pred.pred_instances.keypoint_scores = scores_center

            print("visualize輸出前的 shape:", pred.pred_instances.keypoints.shape)

            # since visualization and inference utilize the same process,
            # the wait time is reduced when a video input is utilized,
            # thereby eliminating the issue of inference getting stuck.
            wait_time = 1e-5 if self._video_input else wait_time

            if num_instances < 0:
                num_instances = len(pred.pred_instances)

            visualization = self.visualizer.add_datasample(
                window_name,
                img,
                data_sample=pred,
                det_data_sample=self._buffer['pose2d_results'],
                draw_gt=False,
                draw_bbox=draw_bbox,
                show=show,
                wait_time=wait_time,
                dataset_2d=self.pose2d_model.model.
                dataset_meta['dataset_name'],
                dataset_3d=self.model.dataset_meta['dataset_name'],
                kpt_thr=kpt_thr,
                num_instances=num_instances)
            results.append(visualization)

            if vis_out_dir:
                img_name = os.path.basename(pred.metainfo['img_path']) \
                    if 'img_path' in pred.metainfo else None
                self.save_visualization(
                    visualization,
                    vis_out_dir,
                    img_name=img_name,
                )

        if return_vis:
            return results
        else:
            return []