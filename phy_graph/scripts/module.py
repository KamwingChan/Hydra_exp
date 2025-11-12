#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import time

from phy_graph_lib.inferenceCore import PhysicalInference
from phy_graph.srv import GetProperties, GetPropertiesResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PhysicalInferenceServer:
    def __init__(self):
        rospy.init_node('phy_graph_server')

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            rospy.logerr("OPENAI_API_KEY environment variable not set.")
            return
        
        base_url = os.environ.get("OPENAI_API_BASE")

        self.inference_tool = PhysicalInference(api_key=api_key, base_url=base_url)
        rospy.loginfo("Initialized PhysicalInference client.")
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        self.service = rospy.Service('get_physical_properties', GetProperties, self.handle_get_properties)
        
        rospy.loginfo("Physical inference service is ready.")

    def handle_get_properties(self, req):
        rospy.loginfo(f"Received request for label: {req.label}")
        
        # Start timing
        start_time = time.time()

        try:
            # receive and process image
            if req.image.height == 0 or req.image.width == 0:
                rospy.logwarn("Received an empty image.")
                return GetPropertiesResponse()
            
            rospy.loginfo(f"Received RGB image: {req.image.width}x{req.image.height}, encoding: {req.image.encoding}")

            # transform ROS Image to OpenCV image
            try:
                cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
            except Exception as e:
                rospy.logerr(f"Failed to convert image: {e}")
                return GetPropertiesResponse()
            
            # 直接从内存中的图像进行推理，无需保存临时文件
            rospy.loginfo(f"Sending image to VLM for inference...")
            properties = self.inference_tool.get_properties_from_image(cv_image, req.label)

            # 5. 验证响应
            if "error" in properties:
                error_msg = properties['error']
                rospy.logerr(f"Inference failed: {error_msg}")
                rospy.logerr(f"Full properties response: {properties}")
                return GetPropertiesResponse()

            required_fields = ["description", "friction_level", "pushable", "weight_level"]
            missing_fields = [field for field in required_fields if field not in properties]
            if missing_fields:
                rospy.logerr(f"Missing required fields in response: {missing_fields}")
                rospy.logerr(f"Full properties response: {properties}")
                return GetPropertiesResponse()

            # 6. 构建响应
            response = GetPropertiesResponse()
            response.description = properties.get("description", "")
            response.friction_level = properties.get("friction_level", 0)
            response.pushable = bool(properties.get("pushable", 0))
            response.weight_level = properties.get("weight_level", 0)
            
            # Calculate processing time
            end_time = time.time()
            processing_time_ms = int((end_time - start_time) * 1000)
            
            rospy.loginfo(f"✓ VLM inference completed in {processing_time_ms}ms")
            rospy.loginfo(f"  Description: {response.description}")
            rospy.loginfo(f"  Friction: {response.friction_level}, Pushable: {response.pushable}, Weight: {response.weight_level}")
            
            return response

        except Exception as e:
            rospy.logerr(f"An error occurred in the service handler: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return GetPropertiesResponse()

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        server = PhysicalInferenceServer()
        server.run()
    except rospy.ROSInterruptException:
        pass
