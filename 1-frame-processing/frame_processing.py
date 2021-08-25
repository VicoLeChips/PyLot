"""
This python file show an exemple of the processing of one frame
        
Functions :
    - Main : contains all frame processing steps

"""

__authors__ = ("BABIN Victor",
               "DEWATRE Pierre",)
__date__ = "23/08/2021"

# Secure importation
try:
    # Data analysis library that allow us to create the transition table
    from LaneFollowing import *
    import cv2
except ImportError:
    print('ImportError for LaneFollowing')

def main():
    frame = cv2.imread('2021-08-19-123009.jpg')
    show_image("Initial Frame", frame)

    #Computation + display
    edges = detect_edges(frame)
    show_image("Extracted Edges", edges)

    #Computation + display
    cropped_edges = cut_top_half(edges)
    show_image("Top Half Deleted", cropped_edges)

    #Computation + display
    line_segments = detect_line_segments(cropped_edges)
    print(line_segments)

    #Computation + display
    lane_lines = detect_lane(frame)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("Lane Lines", lane_lines_image)

    #Computation + display
    steering_angle = compute_steering_angle(lane_lines_image, lane_lines)
    lane_lines_image = display_heading_line(lane_lines_image, steering_angle)
    show_image("Red Heading Line", lane_lines_image)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute only if run as a script
    main()