import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self._bridge = CvBridge()
        self.latest_rgb = None
        self._sub = self.create_subscription(
            Image,
            "/depth_cam/rgb/image_raw",
            self._image_callback,
            qos_profile_sensor_data)

    def _image_callback(self, msg):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
