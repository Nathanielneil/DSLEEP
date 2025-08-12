"""
疲劳检测系统主程序 - Ubuntu 20.04兼容版本
支持实时摄像头检测和GUI界面
"""

import sys
import os
import cv2
import time
import logging
from pathlib import Path

# 设置环境变量，解决Qt库冲突和Ubuntu显示问题
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*.debug=false'
os.environ['QT_X11_NO_MITSHM'] = '1'  # 解决X11共享内存问题
os.environ['QT_XCB_GL_INTEGRATION'] = ''  # 禁用OpenGL集成

# 解决OpenCV和PySide2的Qt插件冲突
# 移除OpenCV的Qt插件路径，优先使用系统Qt
if 'QT_PLUGIN_PATH' in os.environ:
    qt_paths = os.environ['QT_PLUGIN_PATH'].split(':')
    # 过滤掉OpenCV相关的Qt插件路径
    filtered_paths = [p for p in qt_paths if 'cv2' not in p and 'opencv' not in p]
    if filtered_paths:
        os.environ['QT_PLUGIN_PATH'] = ':'.join(filtered_paths)
    else:
        del os.environ['QT_PLUGIN_PATH']

# 确保正确的显示设置
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# 确保模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # 优先尝试PySide6
    try:
        from PySide6 import QtWidgets
        from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QImage, QPixmap, QFont
        GUI_AVAILABLE = True
        print("使用PySide6 GUI框架")
    except ImportError:
        # 备选PySide2
        from PySide2 import QtWidgets
        from PySide2.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit
        from PySide2.QtCore import QTimer, Qt
        from PySide2.QtGui import QImage, QPixmap, QFont
        GUI_AVAILABLE = True
        print("使用PySide2 GUI框架")
except ImportError as e:
    print(f"警告：GUI模块导入失败: {e}")
    print("将运行命令行版本")
    GUI_AVAILABLE = False

import frame_processor

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FatigueDetectionApp:
    """疲劳检测应用主类"""
    
    def __init__(self, use_gui=True):
        self.use_gui = use_gui and GUI_AVAILABLE
        self.cap = None
        self.running = False
        
        # 疲劳检测参数
        self.EYE_THRESH = 0.3
        self.CLOSED_EYE_THRESH = 0.27
        self.MIN_BLINK_FRAMES = 2
        self.MAX_BLINK_FRAMES = 7
        self.CLOSED_EYE_FRAMES = 8
        self.EYE_CONFIRM_FRAMES = 2
        
        # 哈欠检测参数
        self.MOUTH_THRESH = 0.6
        self.MIN_YAWN_FRAMES = 15
        self.YAWN_INTERVAL = 60
        self.MAX_YAWN_COUNT = 3
        
        # 状态变量
        self.eye_closed_confirm = 0
        self.eye_open_confirm = 0
        self.last_eye_state = 'open'
        self.COUNTER = 0
        self.mCOUNTER = 0
        self.TOTAL = 0
        self.mTOTAL = 0
        
        # 时间记录
        self.start_time = time.time()
        self.last_reset_time = time.time()
        self.blink_times = []
        self.yawn_times = []
        self.last_log_time = time.time()
        
        # EAR历史记录
        self.last_ear_values = []
        self.EAR_HISTORY_LENGTH = 5
        self.min_valid_ear = 0.2
        
        if self.use_gui:
            self.init_gui()
        else:
            print("疲劳检测系统 - 命令行版本")
            print("按 'q' 退出")
    
    def init_gui(self):
        """初始化GUI界面"""
        try:
            self.app = QApplication(sys.argv)
            self.window = QMainWindow()
            self.window.setWindowTitle("疲劳检测系统 - Ubuntu版")
            self.window.setGeometry(100, 100, 1000, 700)
            
            # 创建中央窗口部件
            central_widget = QtWidgets.QWidget()
            self.window.setCentralWidget(central_widget)
            
            # 创建布局
            main_layout = QHBoxLayout(central_widget)
            
            # 左侧视频显示区域
            video_layout = QVBoxLayout()
            self.video_label = QLabel("请点击'打开摄像头'开始检测")
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
            self.video_label.setAlignment(Qt.AlignCenter)
            
            # 摄像头控制按钮
            self.camera_button = QPushButton("打开摄像头")
            self.camera_button.clicked.connect(self.toggle_camera)
            
            video_layout.addWidget(self.video_label)
            video_layout.addWidget(self.camera_button)
            
            # 右侧信息显示区域
            info_layout = QVBoxLayout()
            
            # 状态标签
            self.status_label = QLabel("当前状态：等待开始")
            self.status_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.status_label.setStyleSheet("color: blue; padding: 10px;")
            
            self.blink_label = QLabel("眨眼次数：0")
            self.yawn_label = QLabel("哈欠次数：0")
            self.feature_label = QLabel("面部特征：未检测")
            self.fatigue_label = QLabel("疲劳程度：正常")
            
            # 日志显示
            self.log_text = QTextEdit()
            self.log_text.setMaximumHeight(200)
            self.log_text.setReadOnly(True)
            
            info_layout.addWidget(self.status_label)
            info_layout.addWidget(self.blink_label)
            info_layout.addWidget(self.yawn_label)
            info_layout.addWidget(self.feature_label)
            info_layout.addWidget(self.fatigue_label)
            info_layout.addWidget(QLabel("日志信息："))
            info_layout.addWidget(self.log_text)
            
            # 添加到主布局
            main_layout.addLayout(video_layout, 2)
            main_layout.addLayout(info_layout, 1)
            
            # 设置定时器
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.setInterval(33)  # 约30FPS
            
            logger.info("GUI界面初始化完成")
            
        except Exception as e:
            logger.error(f"GUI初始化失败: {e}")
            self.use_gui = False
    
    def log_message(self, message):
        """记录日志消息"""
        current_time = time.strftime("%H:%M:%S", time.localtime())
        log_msg = f"[{current_time}] {message}"
        
        if self.use_gui:
            self.log_text.append(log_msg)
        else:
            print(log_msg)
        
        logger.info(message)
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """启动摄像头"""
        try:
            # Ubuntu下尝试多种摄像头访问方式
            camera_backends = [
                (cv2.CAP_V4L2, "V4L2"),  # Linux Video4Linux2
                (cv2.CAP_GSTREAMER, "GStreamer"),  # GStreamer
                (cv2.CAP_FFMPEG, "FFmpeg"),  # FFmpeg
                (cv2.CAP_ANY, "Any")  # 任意后端
            ]
            
            self.cap = None
            for backend, name in camera_backends:
                try:
                    self.log_message(f"尝试使用 {name} 后端打开摄像头...")
                    self.cap = cv2.VideoCapture(0, backend)
                    if self.cap.isOpened():
                        self.log_message(f"成功使用 {name} 后端")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except Exception as e:
                    self.log_message(f"{name} 后端失败: {e}")
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                self.log_message("错误：无法打开摄像头，请检查：")
                self.log_message("1. 摄像头是否连接")
                self.log_message("2. 用户是否在video组: sudo usermod -a -G video $USER")
                self.log_message("3. 设备权限: ls -l /dev/video*")
                return False
            
            # 设置摄像头缓冲区大小（减少延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 验证设置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.log_message(f"摄像头参数: {actual_width}x{actual_height} @ {actual_fps:.1f}FPS")
            
            # 测试读取一帧
            ret, test_frame = self.cap.read()
            if not ret:
                self.log_message("警告：无法读取摄像头帧")
                self.cap.release()
                return False
            
            self.running = True
            if self.use_gui:
                self.camera_button.setText("关闭摄像头")
                self.timer.start()
            
            self.log_message("摄像头已启动")
            return True
            
        except Exception as e:
            self.log_message(f"启动摄像头失败: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop_camera(self):
        """停止摄像头"""
        self.running = False
        if self.use_gui:
            self.timer.stop()
            self.camera_button.setText("打开摄像头")
            self.video_label.setText("摄像头已关闭")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.log_message("摄像头已关闭")
    
    def update_frame(self):
        """更新帧处理"""
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.log_message("读取帧失败")
            return
        
        try:
            # 处理帧
            processed_frame, eye_ratio, mouth_ratio, label_list = frame_processor.process(frame)
            
            # 更新疲劳状态
            self.update_fatigue_status(eye_ratio, mouth_ratio)
            
            # 更新行为检测
            self.update_behavior_detection(label_list)
            
            # 显示处理后的帧
            if self.use_gui:
                self.display_frame(processed_frame)
            
        except Exception as e:
            self.log_message(f"帧处理错误: {e}")
    
    def display_frame(self, frame):
        """在GUI中显示帧"""
        try:
            # 确保帧格式正确
            if frame is None or len(frame.shape) != 3:
                return
                
            height, width, channels = frame.shape
            
            # 转换BGR到RGB（OpenCV使用BGR格式）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建QImage
            bytes_per_line = channels * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            # 获取标签当前大小并缩放
            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.log_message(f"显示帧错误: {e}")
            # 在错误时显示错误信息
            self.video_label.setText(f"视频显示错误: {str(e)}")
    
    def update_fatigue_status(self, eye_ratio, mouth_ratio):
        """更新疲劳状态"""
        current_time = time.time()
        
        # 定期重置计数器（5分钟）
        if current_time - self.last_reset_time >= 300:
            self.blink_times = []
            self.yawn_times = []
            self.last_reset_time = current_time
            self.log_message("计数器已重置")
        
        # 清理超时记录
        self.blink_times = [t for t in self.blink_times if current_time - t <= 300]
        self.yawn_times = [t for t in self.yawn_times if current_time - t <= self.YAWN_INTERVAL]
        
        # 眼睛状态检测
        if eye_ratio < self.min_valid_ear or eye_ratio == 0:
            self.reset_eye_state()
            return
        
        # 更新EAR历史记录
        self.last_ear_values.append(eye_ratio)
        if len(self.last_ear_values) > self.EAR_HISTORY_LENGTH:
            self.last_ear_values.pop(0)
        
        valid_ears = [e for e in self.last_ear_values if e >= self.min_valid_ear]
        if not valid_ears:
            return
        
        avg_ear = sum(valid_ears) / len(valid_ears)
        
        # 眼睛状态判断
        if avg_ear <= self.CLOSED_EYE_THRESH:
            self.eye_closed_confirm += 1
            self.eye_open_confirm = 0
            
            if self.eye_closed_confirm >= self.EYE_CONFIRM_FRAMES:
                if self.last_eye_state == 'open':
                    self.COUNTER = 1
                    self.last_eye_state = 'closed'
                else:
                    self.COUNTER += 1
        else:
            self.eye_open_confirm += 1
            self.eye_closed_confirm = 0
            
            if self.eye_open_confirm >= self.EYE_CONFIRM_FRAMES:
                if self.last_eye_state == 'closed':
                    if self.MIN_BLINK_FRAMES <= self.COUNTER <= self.MAX_BLINK_FRAMES:
                        self.blink_times.append(current_time)
                        self.TOTAL += 1
                
                self.COUNTER = 0
                self.last_eye_state = 'open'
        
        # 哈欠检测
        if mouth_ratio > self.MOUTH_THRESH:
            self.mCOUNTER += 1
        else:
            if self.mCOUNTER >= self.MIN_YAWN_FRAMES:
                self.yawn_times.append(current_time)
                self.log_message(f"检测到哈欠 - MAR: {mouth_ratio:.3f}")
            self.mCOUNTER = 0
        
        # 判断疲劳状态
        is_fatigue = False
        fatigue_signs = []
        
        if avg_ear <= self.CLOSED_EYE_THRESH:
            is_fatigue = True
            fatigue_signs.append(f"EAR值过低({avg_ear:.3f})")
        
        recent_yawns = len(self.yawn_times)
        if recent_yawns >= self.MAX_YAWN_COUNT:
            is_fatigue = True
            fatigue_signs.append(f"频繁哈欠({recent_yawns}次)")
        
        # 更新GUI显示
        if self.use_gui:
            self.blink_label.setText(f"眨眼次数：{len(self.blink_times)}")
            self.yawn_label.setText(f"哈欠次数：{len(self.yawn_times)}")
            
            if is_fatigue:
                status_text = "警告：检测到疲劳！"
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 10px;")
                
                fatigue_text = "疲劳特征：\\n" + "\\n".join(fatigue_signs)
                self.fatigue_label.setText(fatigue_text)
                self.fatigue_label.setStyleSheet("color: red;")
            else:
                self.status_label.setText(f"当前状态：清醒\\nEAR值：{avg_ear:.3f}")
                self.status_label.setStyleSheet("color: green; padding: 10px;")
                self.fatigue_label.setText("疲劳程度：正常")
                self.fatigue_label.setStyleSheet("color: green;")
    
    def update_behavior_detection(self, label_list):
        """更新行为检测显示"""
        if not self.use_gui:
            return
        
        if label_list:
            # 转换标签为中文
            status_texts = []
            for label in label_list:
                if label in ["Closed eye", "closed_eye"]:
                    status_texts.append("眼睛闭合")
                elif label in ["Opened eye", "open_eye"]:
                    status_texts.append("眼睛张开")
                elif label in ["Yawn", "open_mouth"]:
                    status_texts.append("打哈欠中")
                elif label in ["No-yawn", "closed_mouth"]:
                    status_texts.append("嘴巴闭合")
                else:
                    # 添加对未知标签的处理
                    status_texts.append(f"检测到: {label}")
            
            if status_texts:
                behavior_text = "面部状态：\\n" + "\\n".join(status_texts)
                self.feature_label.setText(behavior_text)
                
                if "眼睛闭合" in status_texts or "打哈欠中" in status_texts:
                    self.feature_label.setStyleSheet("color: red;")
                else:
                    self.feature_label.setStyleSheet("color: green;")
        else:
            self.feature_label.setText("面部特征：未检测到")
            self.feature_label.setStyleSheet("color: black;")
    
    def reset_eye_state(self):
        """重置眼睛状态"""
        self.eye_closed_confirm = 0
        self.eye_open_confirm = 0
        self.COUNTER = 0
        self.last_eye_state = 'open'
        self.last_ear_values = []
    
    def run_console_mode(self):
        """运行命令行模式"""
        if not self.start_camera():
            return
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 处理帧
                processed_frame, eye_ratio, mouth_ratio, label_list = frame_processor.process(frame)
                
                # 更新疲劳状态
                self.update_fatigue_status(eye_ratio, mouth_ratio)
                
                # 显示结果
                cv2.imshow('Fatigue Detection', processed_frame)
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
    
    def run(self):
        """运行应用程序"""
        if self.use_gui:
            self.window.show()
            return self.app.exec_()
        else:
            self.run_console_mode()
            return 0

def main():
    """主函数"""
    print("疲劳检测系统 - Ubuntu 20.04版本")
    print("=" * 50)
    
    # 检查参数
    use_gui = True
    if len(sys.argv) > 1 and sys.argv[1] == '--console':
        use_gui = False
    
    try:
        app = FatigueDetectionApp(use_gui=use_gui)
        return app.run()
        
    except Exception as e:
        logger.error(f"应用程序运行失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())