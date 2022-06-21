from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class Mydataset(data.Dataset):
  num_classes = 1  # airplane
  # default_resolution = [3840, 2160]
  default_resolution = [512, 512]
  # 均值和方差记得改为自己数据集的均值和方差
  mean = np.array([0.92909976, 0.93355255, 0.93746408],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.13014796, 0.11955598, 0.10579153],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(Mydataset, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'plane')
    self.img_dir = os.path.join(self.data_dir, '{}2022'.format(split))
    if split == 'val':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'plane_{}2022.json').format(split)
    else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'plane_{}2022.json').format(split)
    self.max_objs = 128
    self.class_name = [
      '__background__', 'airplane']
    self._valid_ids = [1] # 有效类别索引
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)} # 类别字典
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)] # 类别颜色
    self._data_rng = np.random.RandomState(123) # 随机种子
    # 特征变量_eig_val
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    #特征变量_eig_vec
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split  # split 训练还是测试的指示指标
    self.opt = opt # 配置文件

    print('==> initializing coco 2022 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path) # 加载数据集
    self.images = self.coco.getImgIds() #获取图像id
    self.num_samples = len(self.images) # 得到图像数目

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x): # 转为小数点后两位
    return float("{:.2f}".format(x))
  
  # 转换为评价格式
  def convert_eval_format(self, all_bboxes): 
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  #保存结果
  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  # 评价结果
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

