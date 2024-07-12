#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
import time
import numpy as np

class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

class SolutionNode(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        ### Subscriber to the image topic
        self.subscriber = self.create_subscription(Image,"/prius/front_camera/image_raw",self.callback,10)
        ### Publisher to the control topic
        self.publisher = self.create_publisher(Control, "/prius/control", qos_profile=10)
        self.fps_counter = FPSCounter()
        
        self.bridge = CvBridge()
        self.command = Control()

        self.previous_frame_lines = None
        
        # Parameters for PID controller
        self.target_lane_position = 320  # Target lane position (center of the image)
        self.kp = 0.5  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.1  # Derivative gain
        self.prev_error = 0
        self.integral = 0   
        # Throttle control parameters
        self.max_throttle = 1.0
        self.min_throttle = 0.0
        self.max_brake = 1.0
        self.min_brake = 0.0
        self.throttle_sensitivity = 0.5
        self.brake_sensitivity = 0.5
        self.target_speed = 30.0  # Adjust the target speed as needed
        self.speed_error_integral = 0.0
        self.previous_speed_error = 0.0     
        # Initialize variables for speed calculation
        self.last_speed_update_time = time.time()  # Initialize last speed update time
        self.current_speed = 0.0  # Initialize current speed
        self.distance_traveled = 0.0  # Initialize distance traveled

    def roi(self, image):
        # Convert image to grayscale if needed (assuming the input image is in BGR format)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Define vertices of the region of interest polygon
        vertices = np.array([[(0, image.shape[0]), 
                            (image.shape[1] // 2 - 50, image.shape[0] // 2 + 50),
                            (image.shape[1] // 2 + 50, image.shape[0] // 2 + 50), 
                            (image.shape[1], image.shape[0])]],
                            dtype=np.int32)

        # Create a mask using the vertices
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, 255)

        # Apply the mask to the Canny edge-detected image
        masked_edges = cv2.bitwise_and(edges, mask)

        return masked_edges


    def houghlines(self, masked_edges):
        # Perform Hough line transform
        lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=5)
        
        # Check if lines are found
        if lines is not None:
            # Convert lines to a list of tuples for easier processing
            lines = [line[0] for line in lines]

        return lines

    def draw_lines(self, image, lines):
        lines_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line.reshape(4)  # Extract the coordinates from the array
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        else:
            print("No lines detected.")

        return lines_image

    def calculate_lane_position(self, lines):
        # Calculate the average x-coordinate of lane lines
        if lines is not None:
            x1, _, x2, _ = lines[0][0]
            lane_position = (x1 + x2) // 2
        else:
            # If no lines detected, assume the car is centered
            lane_position = self.target_lane_position
        
        return lane_position

    def calculate_steering_angle(self, lane_position):
        # PID control
        error = self.target_lane_position - lane_position
        self.integral += error
        derivative = error - self.prev_error
        steering_angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        return steering_angle
    
    def throttle_control(self, steering_angle, current_speed):
        # Calculate speed error
        speed_error = self.target_speed - current_speed
        self.speed_error_integral += speed_error

        # PID control for throttle
        throttle = self.throttle_sensitivity * speed_error + self.throttle_sensitivity * self.speed_error_integral

        # PID control for brake
        brake = self.brake_sensitivity * speed_error

        # Ensure throttle and brake values are within limits
        throttle = min(max(throttle, self.min_throttle), self.max_throttle)
        brake = min(max(brake, self.min_brake), self.max_brake)

        # Adjust throttle and brake based on steering angle to reduce speed during turns
        if abs(steering_angle) > 0.1:
            throttle *= 0.5  # Reduce throttle during turns
            brake = 0.5  # Apply moderate brake during turns

        # Return throttle and brake values
        return throttle, brake


    def update_speed(self):
        # Calculate time since last speed update
        current_time = time.time()
        time_elapsed = current_time - self.last_speed_update_time
        
        # Simulate speed calculation (replace with actual speed calculation)
        # For example, you might read speed data from a sensor or calculate based on wheel rotation
        simulated_speed = 5.0  # Simulated speed in m/s
        
        # Update distance traveled
        self.distance_traveled += simulated_speed * time_elapsed
        
        # Update current speed
        self.current_speed = simulated_speed
        
        # Update last speed update time
        self.last_speed_update_time = current_time

    def get_current_speed(self):
        # Update speed before returning current speed
        self.update_speed()
        
        return self.current_speed

    
    def draw_fps(self, img):
        self.fps_counter.step()
        fps = self.fps_counter.get_fps()
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img

    def callback(self, msg: Image):
        # Convert ROS image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        masked_edges = self.roi(cv_image)
        lines = self.houghlines(masked_edges)
        lines_image = self.draw_lines(cv_image, lines)
        # Calculate steering angle
        lane_position = self.calculate_lane_position(self.previous_frame_lines)
        steering_angle = self.calculate_steering_angle(lane_position)
        
        # Calculate current speed (you need to implement this part)
        current_speed = self.get_current_speed()
        
        # Calculate throttle and brake based on steering angle and current speed
        throttle, brake = self.throttle_control(steering_angle, current_speed)
        
        # Create Control message and set throttle and steering angle
        control_msg = Control()
        control_msg.throttle = throttle
        control_msg.steer = steering_angle
        
        # Publish control message
        try:
            self.publisher.publish(control_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing control message: {e}")
        
        # Draw FPS on the image
        cv_image = self.draw_fps(cv_image)
        
        # Show image
        cv2.imshow("prius_front", cv_image)
        cv2.waitKey(1)
        cv2.imshow("lines", lines_image)
        cv2.waitKey(1)
        
        



def main():
    rclpy.init()
    node = SolutionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
