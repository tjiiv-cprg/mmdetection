import os.path as osp

import numpy as np
import mmcv
from viseval import eval_det
from .custom import CustomDataset

from .builder import DATASETS


@DATASETS.register_module()
class VisDroneDataset(CustomDataset):

    CLASSES = ('pedestrian', 'person', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 label_prefix=None):
        self.label_prefix = label_prefix
        self.labels = []
        super(VisDroneDataset, self).__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt)

    @staticmethod
    def open_label_file(path):
        label = np.genfromtxt(path, delimiter=',', dtype=np.int64)
        num_dims = len(label.shape)
        if num_dims == 1:
            label = label.reshape(1, -1)
        return label

    def load_annotations(self, ann_file):
        filenames = open(ann_file).readlines()
        data_infos = []
        for filename in filenames:
            filename_raw = filename.strip()
            img_path = osp.join(self.img_prefix, filename_raw + '.jpg')
            img = mmcv.imread(img_path)
            height, width = img.shape[:2]

            info = dict(filename=filename_raw + '.jpg', width=width, height=height)
            data_infos.append(info)
            if self.label_prefix is not None:
                label = self.open_label_file(
                    osp.join(self.label_prefix, filename_raw + '.txt'))
                self.labels.append(label)
        return data_infos

    def get_ann_info(self, idx):
        return self._parse_ann_info(idx)

    def get_cat_ids(self, idx):
        raise NotImplementedError
        # return self._parse_ann_info(idx)['labels'].tolist()

    def _parse_ann_info(self, idx):
        label = self.labels[idx]
        if label.shape[0]:
            x1y1 = label[:, 0:2]
            wh = label[:, 2:4]
            x2y2 = x1y1 + wh

            # data_info = self.data_infos[idx]
            # width = data_info['width']
            # height = data_info['height']
            # x1y1 = x1y1.clip(min=0)
            # x2y2[:, 0] = x2y2[:, 0].clip(max=width)
            # x2y2[:, 1] = x2y2[:, 1].clip(max=height)
            # wh = x2y2 - x1y1

            _gt_bboxes = np.concatenate((x1y1, x2y2), axis=1).astype(np.float32)
            _gt_labels = label[:, 5]
            _gt_truncation = label[:, 6]
            _gt_occlusion = label[:, 7]

            gt_mask = np.logical_and(label[:, 4], label[:, 5])
            ignore_mask = np.logical_not(gt_mask)

            size_mask = np.min(wh > 0, axis=1)
            class_mask = _gt_labels < 11  # remove "other" category

            gt_mask = np.logical_and(gt_mask, size_mask)
            gt_mask = np.logical_and(gt_mask, class_mask)

            ignore_mask = np.logical_and(ignore_mask, size_mask)

            gt_bboxes = _gt_bboxes[gt_mask]
            gt_labels = _gt_labels[gt_mask]
            gt_truncation = _gt_truncation[gt_mask]
            gt_occlusion = _gt_occlusion[gt_mask]
            gt_bboxes_ignore = _gt_bboxes[ignore_mask]

        else:
            gt_bboxes = np.empty((0, 4), dtype=np.float32)
            gt_labels = np.empty(0, dtype=np.int64)
            gt_bboxes_ignore = np.empty((0, 4), dtype=np.float32)
            gt_truncation = np.empty(0, dtype=np.int32)
            gt_occlusion = np.empty(0, dtype=np.int32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            truncation=gt_truncation,
            occlusion=gt_occlusion)

        return ann

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        results = self.format_results(results)
        heights = [info['height'] for info in self.data_infos]
        widths = [info['width'] for info in self.data_infos]
        ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500 = eval_det(
            self.labels, results, heights, widths)
        eval_res = dict(
            ap_all=float(ap_all),
            ap_50=float(ap_50),
            ap_75=float(ap_75),
            ar_1=float(ar_1),
            ar_10=float(ar_10),
            ar_100=float(ar_100),
            ar_500=float(ar_500)
        )
        return eval_res

    def format_results(self, results, **kwargs):
        results_out = []
        for result in results:
            category = [i
                        for i, det_per_class in enumerate(result)
                        for _ in det_per_class]
            category = np.array(category).reshape(-1, 1)
            result_out = np.concatenate(result, axis=0)
            result_out = np.concatenate((result_out, category), axis=1)
            # sort by scoring in descending order
            result_out = result_out[result_out[:, 4].argsort()[::-1]]
            # x1y1x2y2 to x1y1wh
            result_out[:, 2:4] -= result_out[:, :2]
            results_out.append(result_out)
        return results_out
