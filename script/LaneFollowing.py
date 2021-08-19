import cv2
import numpy as np
import logging
import math

def show_image(name, image):
    cv2.imshow(name,image)
    cv2.waitKey(0)

frame = cv2.imread('/home/pi/PyLot/frame.jpg')
#show_image("Initial Frame", frame)


def detect_edges(frame):
    
    # Isolates blue colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #show_image("HSV Frame", hsv)
    lower_blue = np.array([100, 100, 40]) #Lower spectrum bound, Saturation, Value
    upper_blue = np.array([140, 255, 255]) #Upper spectrum bound, Saturation, Value
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #show_image("Blue Mask", mask)

    # Extracts edges of all the blue areas (Canny Edge Detection Algorithm)
    edges = cv2.Canny(mask, 200, 400)

    return edges

edges = detect_edges(frame)
#show_image("Extracted Edges", edges)


def cut_top_half(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges
cropped_edges = cut_top_half(edges)
show_image("Top Half Deleted", cropped_edges)

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

line_segments = detect_line_segments(cropped_edges)
print(line_segments)

"""
for line in line_segments:
    for x1,y1,x2,y2 in line:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)

show_image("Hough Lines Highlighted", frame)
"""

# Takes a line’s slope and intercept, and returns the endpoints of the line segment
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


# Combine lanes lines
def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    
    # create 2 groups
    left_fit = []
    right_fit = []

    boundary = 1/3  # boundary constant to quick filter lines
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on right 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            """ 
            vertical line skipping (infinite slope issue (could inverse x and y coordinates OR represent the
            line segments in polar coordinates and then averaging angles and distance to the origin)) 
            """ 
            if x1 == x2: 
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1) # least squares polynomial fit
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))


    """take the average of the slopes and intercepts of the line segments to get the slopes 
       and intercepts of left and right lane lines
    """
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)

    return lane_lines

# Given a video frame as input, returns the coordinates of (up to) two lane lines
def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = cut_top_half(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines

# Plot the lane lines on top of the original video frame
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

lane_lines = detect_lane(frame)
lane_lines_image = display_lines(frame, lane_lines)
show_image("Lane Lines", lane_lines_image)



""" Find the steering angle based on lane line coordinate
    We assume that camera is calibrated to point to dead center
"""
def compute_steering_angle(frame, lane_lines):

    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    return steering_angle


# Display red heading line
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
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
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

steering_angle = compute_steering_angle(lane_lines_image, lane_lines)
lane_lines_image = display_heading_line(lane_lines_image, steering_angle)
show_image("Red Heading Line", lane_lines_image)

cv2.destroyAllWindows()
