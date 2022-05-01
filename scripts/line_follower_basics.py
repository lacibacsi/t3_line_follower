#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

# own class to control robot via Twist messages
from move_robot import MoveKobuki


class LineFollower(object):

    # class constants
    # controls the crop size of the image
    # image is 1920 x 1080 for T3
    DESCENTRE = 450  # 160
    ROWS_TO_WATCH = 180  # 300
    UPPER_COLOR_MASK = [50, 255, 255]
    LOWER_COLOR_MASK = [20, 100, 100]

    def __init__(self):

        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.movekobuki_object = MoveKobuki()   # movement control class for the robot

    def camera_callback(self, data):

        try:
            # We select bgr8 because its the OpneCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(
                data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # the below part reads the image dimensions, crops it for faster processing
        # the image then is converted to HSV and yellow filtered / masked then converted to binary black and white
        # the third step is to find the centroid of the white part - aka. the line to follow
        # finally the robot is moved towards the centroid with a fixed speed

        # course notes:
        # We get image dimensions and crop the parts of the image we don't need
        # Bear in mind that because the first value of the image matrix is start and second value is down limit.
        # Select the limits so that it gets the line not too close and not too far, and the minimum portion possible
        # To make process faster.

        height, width, channels = cv_image.shape
        crop_img = cv_image[(height)/2-self.DESCENTRE:(height) /
                            2+self.ROWS_TO_WATCH][1:width]
        # crop_img = cv_image[(height)/2-self.DESCENTRE:(height) /
        #                    2+(self.DESCENTRE+self.ROWS_TO_WATCH)][1:width]
        #crop_img = cv_image[100:400][1:width]

        # Convert from RGB to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        # Define the Yellow Colour in HSV
        #RGB [[[222,255,0]]]
        #BGR [[[0,255,222]]]
        # To know which color to track in HSV, Put in BGR. Use ColorZilla to get the color registered by the camera

        lower_yellow = np.array(self.LOWER_COLOR_MASK)
        upper_yellow = np.array(self.UPPER_COLOR_MASK)

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate centroid of the blob of binary image using ImageMoments
        m = cv2.moments(mask, False)
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cy, cx = height/2, width/2

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(crop_img, crop_img, mask=mask)

        # Draw the centroid in the resultut image
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
        cv2.circle(res, (int(cx), int(cy)), 10, (0, 0, 255), -1)

        # showing all images
        cv2.imshow("Original", cv_image)
        cv2.imshow("HSV", hsv)
        cv2.imshow("MASK", mask)
        cv2.imshow("RES", res)

        cv2.waitKey(1)

        # move the robot here
        error_x = cx - width / 2
        twist_object = Twist()
        twist_object.linear.x = 0.1
        twist_object.angular.z = error_x / 5000  # -error_x / 5000
        rospy.loginfo("ANGULAR VALUE SENT===>"+str(twist_object.angular.z))
        # Make it start turning
        self.movekobuki_object.move_robot(twist_object)

    def clean_up(self):
        # maintenance - close the robot control class, so the robot can be stopped
        self.movekobuki_object.clean_class()
        cv2.destroyAllWindows()


def main():

    rospy.init_node('line_following_node', anonymous=True)
    line_follower_object = LineFollower()

    rate = rospy.Rate(5)
    ctrl_c = False

    def shutdownhook():
        # works better than the rospy.is_shut_down()
        line_follower_object.clean_up()
        rospy.loginfo("shutdown time!")
        ctrl_c = True

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        rate.sleep()


if __name__ == '__main__':
    main()
