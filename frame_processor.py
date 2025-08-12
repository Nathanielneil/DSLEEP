"""
集成疲劳检测和YOLO检测的帧处理接口
"""

import cv2
import time
import logging
from fatigue_detector import get_detector as get_fatigue_detector
from yolo_detector import get_detector as get_yolo_detector

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_frame(frame):
    """
    处理单帧图像，执行疲劳检测和YOLO检测
    
    Args:
        frame: 输入图像帧
    
    Returns:
        tuple: (处理后的帧, 眼睛纵横比, 嘴巴纵横比, 检测标签列表)
    """
    if frame is None:
        return None, 0.0, 0.0, []
    
    try:
        # 调整帧尺寸
        frame = cv2.resize(frame, (640, 480))
        
        # 疲劳检测
        fatigue_detector = get_fatigue_detector()
        frame, eye_ratio, mouth_ratio = fatigue_detector.process_frame(frame)
        
        # YOLO检测
        yolo_detector = get_yolo_detector()
        detections = yolo_detector.predict(frame)
        logger.debug(f"YOLO检测结果数量: {len(detections)}")
        
        # 绘制YOLO检测结果
        label_list = []
        for label, confidence, bbox in detections:
            label_list.append(label)
            logger.debug(f"检测到: {label} (置信度: {confidence:.2f})")
            
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签和置信度
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, eye_ratio, mouth_ratio, label_list
        
    except Exception as e:
        logger.error(f"帧处理失败: {e}")
        return frame, 0.0, 0.0, []

# 兼容原接口
def process(frame):
    """兼容原myframe.process接口"""
    return process_frame(frame)