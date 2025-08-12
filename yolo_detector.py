# YOLO检测模块 - Ubuntu兼容版本
"""
YOLO11检测接口，适配Ubuntu 20.04系统
支持面部特征检测（眼睛开合、打哈欠等状态）
"""

import numpy as np
import cv2
import torch
import os
import sys
import logging
from pathlib import Path

# 确保本地modules和utils可以被导入
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 尝试导入本地模块
try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
    from utils.torch_utils import select_device, time_synchronized
    LOCAL_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"本地模块导入失败: {e}")
    LOCAL_MODULES_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, weights_path=None, device=None, img_size=640, conf_thres=0.3, iou_thres=0.45):
        """初始化YOLO检测器"""
        self.weights_path = weights_path or self._find_weights_path()
        self.device = device or self._select_device()
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.model = None
        self.names = []
        self.colors = []
        self.half = False
        self.model_loaded = False
        
        self._load_model()
    
    def _find_weights_path(self):
        """查找权重文件"""
        possible_paths = [
            os.path.join('weights', 'best.pt'),
            'best.pt',
            os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        logger.error("找不到YOLO权重文件")
        logger.error("请确保以下任一路径存在权重文件：")
        for path in possible_paths:
            logger.error(f"  - {path}")
        return None
    
    def _select_device(self):
        """选择计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info("使用GPU进行推理")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU进行推理")
        return device
    
    def _load_model(self):
        """加载YOLO模型"""
        if not self.weights_path or not os.path.exists(self.weights_path):
            logger.error("权重文件不存在，YOLO检测功能将被禁用")
            return
        
        try:
            # 优先使用本地模块加载（原项目风格）
            if LOCAL_MODULES_AVAILABLE:
                logger.info("使用本地模块加载YOLO模型")
                try:
                    # 使用本地的attempt_load
                    self.model = attempt_load(self.weights_path, map_location=self.device)
                    
                    # 设置为评估模式
                    self.model.eval()
                    
                    # 检查图像尺寸
                    self.img_size = check_img_size(self.img_size, s=self.model.stride.max())
                    
                    # 检查是否可以使用半精度
                    self.half = self.device.type != 'cpu'
                    if self.half:
                        try:
                            self.model.half()
                        except:
                            self.half = False
                            logger.warning("半精度设置失败，使用全精度")
                    
                    # 获取类别名称
                    if hasattr(self.model, 'names'):
                        self.names = self.model.names
                    elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                        self.names = self.model.module.names
                    else:
                        # 默认类别名称（基于疲劳检测）
                        self.names = ['Closed eye', 'Opened eye', 'Yawn', 'No-yawn']
                        logger.warning("未能从模型获取类别名称，使用默认名称")
                    
                    # 生成颜色
                    np.random.seed(42)
                    self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
                    
                    self.model_loaded = True
                    logger.info(f"成功使用本地模块加载YOLO模型: {self.weights_path}")
                    logger.info(f"检测类别: {self.names}")
                    return
                    
                except Exception as local_e:
                    logger.warning(f"本地模块加载失败: {local_e}")
            
            # 备选方案：尝试使用ultralytics库加载
            try:
                from ultralytics import YOLO
                logger.info("使用ultralytics库加载YOLO模型")
                self.model = YOLO(self.weights_path)
                self.model_loaded = True
                self.names = self.model.names if hasattr(self.model, 'names') else ['Closed eye', 'Opened eye', 'Yawn', 'No-yawn']
                logger.info(f"成功使用ultralytics加载YOLO模型: {self.weights_path}")
                logger.info(f"检测类别: {self.names}")
                return
            except ImportError:
                logger.warning("ultralytics库不可用，尝试使用torch直接加载")
            except Exception as ultra_e:
                logger.warning(f"ultralytics加载失败: {ultra_e}")
            
            # 最后备选：直接使用torch加载
            logger.info("尝试直接使用torch加载模型")
            if self.weights_path.endswith('.torchscript'):
                self.model = torch.jit.load(self.weights_path, map_location=self.device)
            else:
                self.model = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            
            # 处理不同的模型格式
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'ema' in self.model:
                    self.model = self.model['ema']
            
            # 检查模型是否可用
            if self.model is None:
                raise Exception("模型加载后为None")
            
            # 设置为评估模式
            self.model.eval()
            
            # 检查是否可以使用半精度
            self.half = self.device.type != 'cpu'
            if self.half:
                try:
                    self.model.half()
                except:
                    self.half = False
                    logger.warning("半精度设置失败，使用全精度")
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                self.names = self.model.module.names
            else:
                # 默认类别名称（基于疲劳检测）
                self.names = ['Closed eye', 'Opened eye', 'Yawn', 'No-yawn']
                logger.warning("未能从模型获取类别名称，使用默认名称")
            
            # 生成颜色
            np.random.seed(42)
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
            
            self.model_loaded = True
            logger.info(f"成功加载YOLO模型: {self.weights_path}")
            logger.info(f"检测类别: {self.names}")
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            logger.error("YOLO检测功能将被禁用，但疲劳检测功能仍可正常使用")
            self.model_loaded = False
            # 设置默认类别名称，即使模型加载失败
            self.names = ['Closed eye', 'Opened eye', 'Yawn', 'No-yawn']
            self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        """调整图像尺寸，保持纵横比"""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def preprocess(self, img):
        """图像预处理"""
        # 调整尺寸
        img_processed = self.letterbox(img, new_shape=self.img_size, auto=False)[0]
        
        # 转换格式：HWC -> CHW
        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img_processed = np.ascontiguousarray(img_processed)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img_processed).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255.0  # 归一化到0-1
        
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """非极大值抑制"""
        # 简化版本的NMS，适用于基本检测
        batch_size = prediction.shape[0]
        output = [torch.zeros((0, 6), device=prediction.device)] * batch_size
        
        for xi, x in enumerate(prediction):  # image index, image inference
            # 筛选置信度
            x = x[x[..., 4] > conf_thres]
            
            if not x.shape[0]:
                continue
            
            # 计算置信度分数
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            
            # 按置信度排序
            i = torch.argsort(x[:, 4], descending=True)
            x = x[i]
            box = box[i]
            
            # 简单的NMS
            keep = self.torchvision_nms(box, x[:, 4], iou_thres)
            output[xi] = x[keep][:max_det]
        
        return output
    
    def xywh2xyxy(self, x):
        """转换边界框格式 (center x, center y, width, height) to (x1, y1, x2, y2)"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y
    
    def torchvision_nms(self, boxes, scores, iou_threshold):
        """简化版NMS"""
        try:
            import torchvision
            return torchvision.ops.nms(boxes, scores, iou_threshold)
        except ImportError:
            # 如果没有torchvision，使用简单的实现
            return self.simple_nms(boxes, scores, iou_threshold)
    
    def simple_nms(self, boxes, scores, iou_threshold):
        """简单的NMS实现"""
        keep = []
        idx = torch.argsort(scores, descending=True)
        
        while len(idx) > 0:
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            
            idx = idx[1:]
            ious = self.box_iou(boxes[i:i+1], boxes[idx])
            idx = idx[ious[0] <= iou_threshold]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def box_iou(self, box1, box2):
        """计算IoU"""
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, 0:1], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1:2], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2:3], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3:4], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = area1.unsqueeze(1) + area2 - inter_area
        
        return inter_area / union_area
    
    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """将坐标从img1_shape缩放到img0_shape"""
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords
    
    def clip_coords(self, boxes, img_shape):
        """限制坐标在图像范围内"""
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    
    def predict(self, img):
        """执行预测"""
        if not self.model_loaded:
            logger.warning("模型未加载，跳过YOLO预测")
            return []
        
        try:
            # 如果使用本地模块，使用原项目的预测流程
            if LOCAL_MODULES_AVAILABLE and hasattr(self, 'model') and self.model is not None:
                results = self._predict_local(img)
                logger.debug(f"YOLO本地预测结果: {results}")
                return results
            else:
                # 使用简化的预测流程
                results = self._predict_simple(img)
                logger.debug(f"YOLO简化预测结果: {results}")
                return results
            
        except Exception as e:
            logger.error(f"YOLO预测失败: {e}")
            return []
    
    def _predict_local(self, img):
        """使用本地模块的预测方法（原项目风格）"""
        try:
            # 预处理（使用原项目的letterbox）
            img_processed = self.letterbox(img, new_shape=self.img_size, auto=False)[0]
            img_processed = img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_processed = np.ascontiguousarray(img_processed)
            
            # 转为tensor
            img_tensor = torch.from_numpy(img_processed).to(self.device)
            img_tensor = img_tensor.half() if self.half else img_tensor.float()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                pred = self.model(img_tensor, augment=False)[0]
                
                # 使用本地的non_max_suppression
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            # 后处理
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # 缩放坐标回原图尺寸
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
                    
                    # 处理每个检测结果
                    for *xyxy, conf, cls in reversed(det[:5]):  # 限制最多5个检测结果
                        try:
                            cls_idx = int(cls)
                            if conf > self.conf_thres and cls_idx < len(self.names):
                                # 获取标签名
                                if isinstance(self.names, dict):
                                    label = self.names[cls_idx]
                                elif isinstance(self.names, list):
                                    label = self.names[cls_idx]
                                else:
                                    label = str(cls_idx)
                                results.append((label, float(conf), [int(x) for x in xyxy]))
                        except Exception as e:
                            logger.warning(f"处理检测结果失败: {e}")
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"本地预测失败: {e}")
            return []
    
    def _predict_simple(self, img):
        """简化的预测方法"""
        try:
            # 预处理
            img_tensor = self.preprocess(img)
            
            # 推理
            with torch.no_grad():
                pred = self.model(img_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            # 后处理
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # 缩放坐标回原图尺寸
                    det[:, :4] = self.scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
                    
                    # 处理每个检测结果
                    for *xyxy, conf, cls in det[:5]:  # 限制最多5个检测结果
                        try:
                            cls_idx = int(cls)
                            if conf > self.conf_thres and cls_idx < len(self.names):
                                # 获取标签名
                                if isinstance(self.names, dict):
                                    label = self.names[cls_idx]
                                elif isinstance(self.names, list):
                                    label = self.names[cls_idx]
                                else:
                                    label = str(cls_idx)
                                results.append((label, float(conf), [int(x) for x in xyxy]))
                        except Exception as e:
                            logger.warning(f"处理检测结果失败: {e}")
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"简化预测失败: {e}")
            return []

# 全局检测器实例
_detector = None

def get_detector():
    """获取全局检测器实例"""
    global _detector
    if _detector is None:
        _detector = YOLODetector()
    return _detector

def predict(img):
    """兼容原接口的预测函数"""
    detector = get_detector()
    return detector.predict(img)