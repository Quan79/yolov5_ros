#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
from cv_bridge import CvBridge
import pathlib
from rostopic import get_topic_type
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from detection_msgs.msg import BoundingBox, BoundingBoxes

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# add yolov5 submodule to path
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = pathlib.Path(os.path.relpath(ROOT, pathlib.Path.cwd()))  # relative path

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
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
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

        depth_camera_info_topic = rospy.get_param(
            '~depth_camera_info_topic', '/camera_front/depth/camera_info'
        )
        color_camera_info_topic = rospy.get_param(
            '~color_camera_info_topic', '/camera_front/color/camera_info'
        )

        #camera_info subscribe
        self.camera_info_color_sub = rospy.Subscriber(color_camera_info_topic, CameraInfo, self.camera_info_color_callback)

        #camera_info subscribe
        self.camera_info_depth_sub = rospy.Subscriber(depth_camera_info_topic, CameraInfo, self.camera_info_depth_callback)

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=1
        )
        self.aligned_pub = rospy.Publisher('/yolov5/depth_image',  Image, queue_size=1)
        # Initialize controller publisher
        self.controller_pub = rospy.Publisher(
            rospy.get_param("~controller_topic"), Float64, queue_size=1
        )

        self.cones_pub = rospy.Publisher(
            "/cones_position",  Path, queue_size=1)
        
        self.stop_pub = rospy.Publisher(
            "/stop_sgin", Float64MultiArray, queue_size=1
        )

        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")

        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~prediction"), Image, queue_size=10
            )
        # self.cones_pub = rospy.Publisher(
        #     "/cones_position",  Path, queue_size=1)


        # Initialize CV_Bridge
        self.bridge = CvBridge()
        self.color_image = Image()
        self.depth_image = Image()
        self.camera_info_depth = CameraInfo()
        self.camera_info_color = CameraInfo()

    def depth_callback(self, depth_image_msg):

        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")

        aligned_depth_image = self.align_depth_to_color(depth_image, self.camera_info_depth, self.camera_info_color)
        self.depth_image = aligned_depth_image

        self.aligned_pub.publish(self.bridge.cv2_to_imgmsg(self.depth_image, "passthrough"))


    def camera_info_depth_callback(self, camera_info_depth_msg):
        self.camera_info_depth = camera_info_depth_msg

    def camera_info_color_callback(self, camera_info_color_msg):
        self.camera_info_color = camera_info_color_msg


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

        cones_msg = Path()
        current_time = rospy.Time.now()
        cones_msg.header.stamp = current_time
        cones_msg.header.frame_id = "vehicle_frame"


        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                self.c = int(cls)

                # Calculate the center of the bounding box
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)


                # Add depth information to the bounding box
                if self.depth_image is not None:
                    self.distance_bbc = self.depth_image[int (y_center), int (x_center)]
                    bounding_box.distance = float (self.distance_bbc)

                # Add stop sign output topic 
                stop_sign = Float64MultiArray()
                stop_sign.layout = self.c
                stop_sign.data = self.distance_bbc
                self.stop_pub.publish(stop_sign)

                # if self.c == 0:
                #      if self.distance_bbc < 500:
                #             brk = 0
                #             self.controller_pub.publish(float(brk))

                # Fill in bounding box message
                bounding_box.Class = self.names[self.c]
                bounding_box.probability = conf
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])


                pose_stamped = PoseStamped()
                cones_msg.header.stamp = current_time
                cones_msg.header.frame_id = "vehicle_frame"
                pose_stamped.pose.position.z = bounding_box.distance /1000
                pose_stamped.pose.position.x = (x_center -638.79 )/645.7 * pose_stamped.pose.position.z
                pose_stamped.pose.position.y = (y_center -352.38 )/645.7 * pose_stamped.pose.position.z
                cones_msg.poses.append(pose_stamped)
                #
                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[self.c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(self.c, True))


                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)
        self.cones_pub.publish(cones_msg)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
            self.color_image = im0


    def align_depth_to_color(self, depth_image, camera_info_depth, camera_info_color):
        # 获取相机内参
        K_depth = np.array(camera_info_depth.K).reshape(3, 3)
        K_color = np.array(camera_info_color.K).reshape(3, 3)

        D_depth = np.array(camera_info_depth.D)
        D_color = np.array(camera_info_color.D)

        R_depth = np.array(camera_info_depth.R).reshape(3, 3)
        R_color = np.array(camera_info_color.R).reshape(3, 3)

        P_depth = np.array(camera_info_depth.P).reshape(3, 4)
        P_color = np.array(camera_info_color.P).reshape(3, 4)

        # 计算对齐映射
        map_x, map_y = cv2.initUndistortRectifyMap(K_depth, D_depth, R_depth, K_color, depth_image.shape[::-1], cv2.CV_32FC1)
        aligned_depth_image = cv2.remap(depth_image, map_x, map_y, cv2.INTER_LINEAR)

        return aligned_depth_image

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0



if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))

    rospy.init_node("yolov5", anonymous=True)
    # rospy.Rate(50)
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