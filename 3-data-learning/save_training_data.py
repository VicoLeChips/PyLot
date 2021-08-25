import cv2
import sys
from LaneFollowing import LaneFollower


##############################
# Save training data function 
##############################
#Save image and steering angle in file name
def save_image_and_steering_angle(video_file):
    lane_follower = LaneFollower()
    cap = cv2.VideoCapture(video_file + '.avi')

    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            lane_follower.follow_lane(frame)
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower.curr_steering_angle), frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()



############################
# Main
############################
if __name__ == '__main__':
    save_image_and_steering_angle(sys.argv[1]) #Saving images with steering angle from list of arguments