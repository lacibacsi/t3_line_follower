#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist


class MoveKobuki(object):

    def __init__(self):

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # for storing the last command? where is this used??
        self.last_cmdvel_command = Twist()
        self._cmdvel_pub_rate = rospy.Rate(10)  # 10Hz
        # as apparently using rospy.isshutdown is suboptimal?
        self.shutdown_detected = False

    def move_robot(self, twist_object):
        # just forward the twist to the publisher, no validation here
        self.cmd_vel_pub.publish(twist_object)

    def clean_class(self):
        # Stop Robot - both forward motion and turning is cleared
        twist_object = Twist()
        twist_object.linear.x = 0.0
        twist_object.angular.z = 0.0
        self.move_robot(twist_object)
        self.shutdown_detected = True   # so hookup can kick in

# just an example? this is not really doing anything apart from turning the robot around


def main():
    rospy.init_node('move_robot_node', anonymous=True)

    movekobuki_object = MoveKobuki()
    twist_object = Twist()
    # Make it start turning
    twist_object.angular.z = 0.15

    rate = rospy.Rate(5)

    ctrl_c = False

    def shutdownhook():
        # works better than the rospy.is_shut_down()
        movekobuki_object.clean_class()  # calling cleanup method that will stop the robot
        rospy.loginfo("shutdown time!")
        ctrl_c = True

    rospy.on_shutdown(shutdownhook)

    while not ctrl_c:
        movekobuki_object.move_robot(twist_object)
        rate.sleep()


if __name__ == '__main__':
    main()
