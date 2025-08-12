"""
疲劳检测，检测眼睛和嘴巴的开合程度
支持Ubuntu 20.04系统
"""

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import math
import os
import sys
from threading import Thread
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FatigueDetector:
    def __init__(self, model_path=None):
        """初始化疲劳检测器"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.model_loaded = False
        
        # 设置模型路径
        if model_path is None:
            model_path = self._find_model_path()
        
        self._load_model(model_path)
        
        # 定义眼睛和嘴巴的关键点索引
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        
        # 检测参数
        self.scales = [0.5, 1.0, 1.5]  # 多尺度检测
        self.padding_ratio = 0.1  # 人脸区域扩展比例
    
    def _find_model_path(self):
        """查找人脸关键点模型文件"""
        possible_paths = [
            os.path.join('weights', 'shape_predictor_68_face_landmarks.dat'),
            'shape_predictor_68_face_landmarks.dat',
            os.path.join(os.path.dirname(__file__), 'weights', 'shape_predictor_68_face_landmarks.dat')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        logger.error("找不到人脸关键点模型文件")
        logger.error("请确保以下任一路径存在模型文件：")
        for path in possible_paths:
            logger.error(f"  - {path}")
        return None
    
    def _load_model(self, model_path):
        """加载人脸关键点检测模型"""
        if model_path is None or not os.path.exists(model_path):
            logger.error("模型文件不存在，疲劳检测功能将被禁用")
            return
        
        try:
            self.predictor = dlib.shape_predictor(model_path)
            self.model_loaded = True
            logger.info(f"成功加载人脸关键点模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型出错: {e}")
            self.model_loaded = False
    
    def eye_aspect_ratio(self, eye):
        """计算眼睛纵横比 (EAR)"""
        # 计算眼睛纵向的两组点的欧氏距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # 计算眼睛横向的欧氏距离
        C = dist.euclidean(eye[0], eye[3])
        # 计算眼睛的纵横比
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """计算嘴巴纵横比 (MAR)"""
        # 计算嘴巴纵向的欧氏距离
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        # 计算嘴巴横向的欧氏距离
        C = dist.euclidean(mouth[0], mouth[6])
        # 计算嘴巴的纵横比
        mar = (A + B) / (2.0 * C)
        return mar
    
    def detect_faces_multiscale(self, frame_gray):
        """多尺度人脸检测"""
        faces = []
        for scale in self.scales:
            try:
                # 调整图像尺寸
                if scale != 1.0:
                    height, width = frame_gray.shape
                    new_height, new_width = int(height * scale), int(width * scale)
                    scaled_frame = cv2.resize(frame_gray, (new_width, new_height))
                else:
                    scaled_frame = frame_gray
                
                # 检测人脸
                detected = self.detector(scaled_frame, 0)
                if detected:
                    # 将检测结果转换回原始尺度
                    faces = [(int(rect.left()/scale), int(rect.top()/scale), 
                             int(rect.right()/scale), int(rect.bottom()/scale)) 
                            for rect in detected]
                    break
            except Exception as e:
                logger.warning(f"尺度 {scale} 检测失败: {e}")
                continue
        
        return faces
    
    def process_frame(self, frame):
        """处理单帧图像，返回处理后的图像和检测结果"""
        if not self.model_loaded:
            return frame, 0.0, 0.0
        
        # 预处理图像以提高检测率
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)  # 直方图均衡化
        
        # 多尺度人脸检测
        faces = self.detect_faces_multiscale(frame_gray)
        
        eyear = 0.0
        mouthar = 0.0
        
        if faces:  # 如果检测到人脸
            # 使用最大的人脸
            face_rect = max(faces, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
            left, top, right, bottom = face_rect
            
            # 扩大检测区域
            padding_h = int((bottom - top) * self.padding_ratio)
            padding_w = int((right - left) * self.padding_ratio)
            
            # 确保扩展后的坐标不超出图像范围
            height, width = frame.shape[:2]
            rect = dlib.rectangle(
                max(0, left - padding_w),
                max(0, top - padding_h),
                min(width, right + padding_w),
                min(height, bottom + padding_h)
            )
            
            try:
                # 关键点检测
                shape = self.predictor(frame_gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # 获取眼睛和嘴巴坐标
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                mouth = shape[self.mStart:self.mEnd]
                
                # 计算眼睛和嘴巴的纵横比
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                eyear = (leftEAR + rightEAR) / 2.0
                mouthar = self.mouth_aspect_ratio(mouth)
                
                # 绘制面部特征点和轮廓
                self._draw_landmarks(frame, shape, leftEye, rightEye, mouth)
                
                # 在画面上显示当前的EAR和MAR值
                cv2.putText(frame, f"EAR: {eyear:.2f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mouthar:.2f}", (300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                           
            except Exception as e:
                logger.warning(f"关键点检测失败: {str(e)}")
                return frame, 0.0, 0.0
        
        return frame, eyear, mouthar
    
    def _draw_landmarks(self, frame, shape, leftEye, rightEye, mouth):
        """绘制面部特征点和轮廓"""
        # 绘制所有特征点
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # 绘制眼睛和嘴巴轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

# 全局检测器实例
_detector = None

def get_detector():
    """获取全局检测器实例"""
    global _detector
    if _detector is None:
        _detector = FatigueDetector()
    return _detector

def detfatigue(frame):
    """兼容原接口的疲劳检测函数"""
    detector = get_detector()
    return detector.process_frame(frame)