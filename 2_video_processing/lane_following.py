import cv2
import numpy as np
import logging
import math
import datetime
import sys

############################
# Constant
############################
_SHOW_IMAGE = False

############################
# Lane Follower Class
############################
class LaneFollower(object):

    #Constructor 
    def __init__(self, car=None):
        logging.info('Creating a LaneFollower...')
        self.car = car
        self.curr_steering_angle = 90
    
    # Main entry point of the lane follower
    def follow_lane(self, frame):
        show_image("Original Frame", frame)
        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)
        return final_frame

    #Steering method
    def steer(self, frame, lane_lines):
        logging.debug('Steering...')
        if len(lane_lines) == 0:
            logging.error('No lane lines detected ! Keep going !')
            return frame

        #Steering angle calculation
        new_steering_angle = compute_steering_angle(frame, lane_lines)
        
        #Stabilization
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

        #Applying steering
        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)

        #Heading Red Line
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image("Heading Red Line", curr_heading_image)

        return curr_heading_image


############################
# Frame processing steps
############################

#Lane detection
def detect_lane(frame):
    logging.debug('detecting lane lines...')

    #Edge detection
    edges = detect_edges(frame)
    show_image('Edge Detection', edges)

    #Cropped edge
    cropped_edges = cut_top_half(edges)
    show_image('Cropped Edges', cropped_edges)

    #Line segment
    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image("Line Segments", line_segment_image)

    #Transforming line segment to lane lines
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("Lane Lines", lane_lines_image)

    return lane_lines, lane_lines_image


#Edge detection
def detect_edges(frame):
    #Isolates blue colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("HSV Frame", hsv)
    lower_blue = np.array([90, 100, 40]) #Lower spectrum bound, Saturation, Value
    upper_blue = np.array([150, 255, 255]) #Upper spectrum bound, Saturation, Value
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("Blue Mask", mask)

    #Extracts edges of all the blue areas (Canny Edge Detection Algorithm)
    edges = cv2.Canny(mask, 200, 400)

    return edges


#Cut top half of the frame
def cut_top_half(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    #Focus on bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    #Applying mask
    cv2.fillPoly(mask, polygon, 255)
    show_image("Mask Top Half", mask)
    masked_image = cv2.bitwise_and(canny, mask)

    return masked_image


#Line segment detection
def detect_line_segments(cropped_edges):
    rho = 1  #Precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  #Degree in radian, i.e. 1 degree
    min_threshold = 10  #Minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)

    if line_segments is not None:
        for line_segment in line_segments:
            logging.debug('Line segment detected:')
            logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))

    return line_segments


#Combine lanes lines with slope
def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line segments detected')
        return lane_lines

    height, width, _ = frame.shape

    #Creationg of 2 groups
    left_fit = []
    right_fit = []

    #Quick filtering line on screen position
    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on right 1/3 of the screen

    #Line segment combination
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('Vertical line segment detected: %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    #Left lane slope average
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    #Right lane slope average
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('Lane lines: %s' % lane_lines)

    return lane_lines


#Compute steering angle
def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    #If no lane lines detected
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    #If lane lines detected
    height, width, _ = frame.shape
    #If one signel line detected
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    #If both lines are detected
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    #Find the steering angle, which is angle between navigation direction to end of center line (red heading line)
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  #Angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  #Angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  #Steering angle needed by picar front wheel

    logging.debug('New steering angle: %s' % steering_angle)
    return steering_angle


#Stabilization of the steering angle
def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        #If both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        #If only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Proposed angle: %s | Stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    
    return stabilized_steering_angle


############################
# Utility Functions
############################

#Display line 
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    #Adding weight to line
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


#Display red heading line
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # The steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    #Display line with line_color = red
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    #Adding weight to heading line
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


#Length of line segment
def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


#Show image (window creation)
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


# Takes a line’s slope and intercept, and returns the endpoints of the line segment
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  #Bottom of the frame
    y2 = int(y1 * 1 / 2)  #Make points from middle of the frame down

    #Bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]
