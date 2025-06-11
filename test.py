import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
import cv2
import os
import glob
import json
 
class AnnotatedImagePublisher:
    def __init__(self):
        rospy.init_node('annotated_image_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('/instance_segmentation/visualisation', Image, queue_size=10)
        self.detection_pub = rospy.Publisher('/instance_segmentation/detections', Detection2DArray, queue_size=10)
        self.bridge = CvBridge()
 
        # Path to image folder and annotations
        self.image_dir = '/home/neura_ai/data/object_perception/trained_models/instance_segmentation/yolo/lavera_s/v_1/'

        self.annotation_dir = '/home/neura_ai/Documents/lavera_1/real/train/coco_annotations.json'

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')) +
                glob.glob(os.path.join(self.image_dir, '*.png')))

        if not self.image_paths:
            rospy.logerr(f"No image files found in {self.image_dir}")

        self.index = 0
        self.rate = rospy.Rate(2)
 
    def get_annotations_for_image(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.annotation_dir, base_name + '.json')
 
        if not os.path.exists(json_path):
            rospy.logwarn(f"No annotation found for {base_name}")
            return []
 
        with open(json_path, 'r') as f:
            try:
                annotations = json.load(f)
                return annotations
            except Exception as e:
                rospy.logwarn(f"Failed to load JSON {json_path}: {e}")
                return []
 
    def draw_annotations(self, img, annotations):
        for ann in annotations:
            bbox = ann.get('bbox')
            label = ann.get('label', 'unknown')
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img
 
    def create_detection_message(self, annotations, header):
        detection_array = Detection2DArray()
        detection_array.header = header
 
        for ann in annotations:
            bbox = ann.get('bbox')
            label = ann.get('label', 'unknown')
 
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
 
                det = Detection2D()
                det.header = header
 
                # Add label and score (score is optional)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = label
                hypothesis.score = 1.0  # Use 1.0 for ground truth
 
                det.results.append(hypothesis)
 
                # Set bounding box
                box = BoundingBox2D()
                box.center.x = cx
                box.center.y = cy
                box.size_x = w
                box.size_y = h
 
                det.bbox = box
 
                detection_array.detections.append(det)
 
        return detection_array
 
    def start_publishing(self):
        while not rospy.is_shutdown():
            if self.index >= len(self.image_paths):
                self.index = 0
 
            img_path = self.image_paths[self.index]
            img = cv2.imread(img_path)
 
            if img is None:
                rospy.logwarn("Failed to read image: {}".format(img_path))
                self.index += 1
                continue
 
            annotations = self.get_annotations_for_image(img_path)
            annotated_img = self.draw_annotations(img, annotations)
 
            try:
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera"
 
                # Publish image
                ros_image = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
                ros_image.header = header
                self.image_pub.publish(ros_image)
 
                # Publish detections
                detection_msg = self.create_detection_message(annotations, header)
                self.detection_pub.publish(detection_msg)
 
                rospy.loginfo(f"Published image and {len(annotations)} detections")
 
            except Exception as e:
                rospy.logerr(f"Error converting/publishing image or detections: {e}")
 
            self.index += 1
            self.rate.sleep()
 
if __name__ == '__main__':
    try:
        publisher = AnnotatedImagePublisher()
        publisher.start_publishing()
    except rospy.ROSInterruptException:
        pass