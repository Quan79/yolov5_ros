#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import random
import time
from cv_bridge import CvBridge
from pathlib import Path
from rostopic import get_topic_type
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup        
        
        # Initialize subscriber to Image/CompressedImage topic
        # 初始化订阅输入图像话题，获取topic type， topic name
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        #判断这个input_image_type是不是compressedimage类型的消息
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage" 

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=10
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=10
            )

        #Initialize subscriber to depth image topic 
        input_depth_type, input_depth_topic, _ = get_topic_type(rospy.get_param("~depth_image_topic"), blocking = True)

        self.depth_image_sub = rospy.Subscriber(
            input_depth_topic, Image, self.depth_callback, queue_size=10
        )


        # Initialize prediction publisher 
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )

        # Initialize controller publisher
        self.controller_pub = rospy.Publisher(
            rospy.get_param("~controller_topic"), Float64, queue_size=10
        )

        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~depth_output_topic"), Image, queue_size=10
            )
        
        """ 如果已经存在相机的深度图publisher应该不需要写这一部分，不确定，可能也需要一个publisher发布带框的深度图
        #Initialize depth publisher
        self.publish_depth_image = rospy.get_param("~publish_depth_image")
        if self.publish_depth_image:
            self.depth_pub = rospy.Publisher(
                rospy.get_param("~output_depth_image"), Image, queue_size=10
            )
        """
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.c = None
        
     #深度callback函数
    def depth_callback(self, depth_image):
        try:
            # Convert depth image to numpy array
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            self.depth_image = np.array(cv_depth_image, dtype=np.float64)
            self.depth_image = np.nan_to_num(self.depth_image)  # Replace NaN values with 0
        except CvBridgeError as e:
            rospy.logerr('Error converting depth image: {}'.format(e))
            return  
           

#    def filter(self, x, y, min_val, randnum):
#        distance_list = []
#        #mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
#        #min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
#        #print(box)
#        for i in range(randnum):
#            bias = random.randint(-min_val//4, min_val//4)
#            dist = self.depth_image[int(y + bias), int(x + bias)]
#            if dist:
#                distance_list.append(dist)
#        distance_list = np.array(distance_list)
#        distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
#    #print(distance_list, np.mean(distance_list))
#        distance = np.mean(distance_list)
#        return distance


    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        #取letterbox返回的第一个值，即img，转换成np数组
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]]) 
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        #重新构造一个连续数组img
        img = np.ascontiguousarray(img)

        return img, img0


    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # print(data.header)
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(im)
        # print(im.shape)
        # print(img0.shape)
        # print(img.shape)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            # 将Boundingboxes 恢复到原始图像im0的尺寸
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                self.c = int(cls)

                # Calculate the center of the bounding box
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)

                #min_val = min(abs(xyxy[2] - xyxy[0]), abs(xyxy[3] - xyxy[1]))

                # Add depth information to the bounding box
                if self.depth_image is not None:
                    self.distance_bbc = self.depth_image[int (y_center), int (x_center)]
                #    distance_bbc = self.filter(x_center, y_center, min_val, 24)
                    bounding_box.distance = float (self.distance_bbc)

                if self.c == 0:
                     if self.distance_bbc < 500:
                            brk = 0
                            self.controller_pub.publish(float(brk))

                # Fill in bounding box message
                bounding_box.Class = self.names[self.c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])
 
                #
                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[self.c]} {conf:.2f} {bounding_box.distance}"
                    annotator.box_label(xyxy, label, color=colors(self.c, True))     

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
            self.color_image = im0            
    
        

if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()

    # def countdown_timer(seconds):
    #     while seconds > 0:
    #         time.sleep(1)
    #         seconds -= 1
    #     return seconds

    # # Check stop sign
    # if detector.c == 0:
    #     if detector.distance_bbc < 700:
    #         brk = 0
    #         detector.controller_pub.publish(float(brk))
    #         # seconds = 5
    #         # detector.countdown_timer(seconds)
    #         # if seconds == 0:
    #         #     brk = 1
    #         #     detector.controller_pub.publish(float(brk))
    
    rospy.spin()
