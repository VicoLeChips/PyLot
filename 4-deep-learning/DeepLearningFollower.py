import cv2
import numpy as np
import logging
import math
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model 
from LaneFollowing import LaneFollower


############################
# Constant
############################
_SHOW_IMAGE = False





######################################
# Deep Learning Lane Follower Class
######################################
class DeepLearningLaneFollower(object):

    #Constructor 
    def __init__(self,
                 car=None,
                 model_path='/home/pi/Desktop/PyLot_COPY/4-deep-learning/lane_navigation_final.h5'):
        logging.info('Creating a DeepLearningLaneFollower...')

        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model(model_path, compile = False)  ## force keras 

    # Main entry point of the lane follower
    def follow_lane(self, frame):
        show_image("Initial Frame", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logging.debug("Current steering angle = %d" % self.curr_steering_angle)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

    #Steering method
    def compute_steering_angle(self, frame):
        """ Find the steering angle directly based on video frame
            We assume that camera is calibrated to point to dead center
        """
        keras.backend.set_learning_phase(0)  #Seems to make the model inference with keras run faster
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = self.model.predict(X)[0]
    
        """
        preprocessed = img_preprocess(frame)
        X = np.asarray([preprocessed])
        steering_angle = self.model.predict(X)[0]"""

        logging.debug('New steering angle: %s' % steering_angle)
        return int(steering_angle + 0.5) #Round the nearest integer




############################
# Image processing steps
############################
#Image processing
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]  #Remove top half of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  #Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) #Input image size (200,66) Nvidia model
    image = image / 255 #Normalizing, the processed image becomes black for some reason
    return image




############################
# Utility Functions
############################
#Display Heading Line
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    
    #Adding weight to line
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


#Show image function
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)




############################
# Test Function
############################
#Video simulation
def video_simulation(video_file):
    end_to_end_lane_follower = DeepLearningLaneFollower()
    hand_coded_lane_follower = LaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    #Skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    cv2.resizeWindow('frame',320,240)
    video_type = cv2.VideoWriter_fourcc(*'XVID')
    video_overlay = cv2.VideoWriter("%s_Deep_Learning.avi" % video_file, video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            frame_copy = frame.copy()
            logging.info('Frame %s' % i)
            combo_image1 = hand_coded_lane_follower.follow_lane(frame)
            combo_image2 = end_to_end_lane_follower.follow_lane(frame_copy)

            diff = end_to_end_lane_follower.curr_steering_angle - hand_coded_lane_follower.curr_steering_angle;
            logging.info("Desired=%3d | Model=%3d | Diff=%3d" %
                          (hand_coded_lane_follower.curr_steering_angle,
                          end_to_end_lane_follower.curr_steering_angle,
                          diff))
            
            combo_image2=cv2.resize(combo_image2,(320, 240))
            video_overlay.write(combo_image2)
            cv2.imshow("Hand Coded", combo_image1)
            cv2.imshow("Deep Learning", combo_image2)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()



############################
# Main
############################
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    #Simulating with a video
    video_simulation('/home/pi/Desktop/PyLot_COPY/4-deep-learning/video01')
    