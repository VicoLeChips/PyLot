import logging
import cv2
import datetime
from LaneFollowing import LaneFollower
import picar

############################
# Constant
############################
_SHOW_IMAGE = True




############################
# PyLot Class
############################
class PyLot(object):

    __INITIAL_SPEED = 0
    __SCREEN_WIDTH = 340
    __SCREEN_HEIGHT = 240

    def __init__(self):
        #Camera and wheel initiation
        logging.info('Creating a PyLot...')

        #SunFounder Setup of the car
        picar.setup()

        #Camera Setup
        logging.debug('Set up camera')
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3, self.__SCREEN_WIDTH)
        self.camera.set(4, self.__SCREEN_HEIGHT)

        #Back wheel setup
        logging.debug('Set up back wheels')
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.back_wheels.speed = 0  #Speed Range is 0 (stop) - 100 (fastest)

        #Front wheel setup
        logging.debug('Set up front wheels')
        self.front_wheels = picar.front_wheels.Front_Wheels()
        self.front_wheels.turning_offset = 0  #Calibrate servo to center
        self.front_wheels.turn(90)  #Steering Range is 45 (left) - 90 (center) - 135 (right)

        #Running lane follower
        self.lane_follower = LaneFollower(self)

        cv2.resizeWindow('frame',340,240)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.video_lane = self.create_video_recorder('car_video_lane%s.avi' % datestr)
        
        logging.info('Created a PyLot')


    def create_video_recorder(self, path):
        return cv2.VideoWriter(path, self.fourcc, 10.0, (self.__SCREEN_WIDTH, self.__SCREEN_HEIGHT))


    def __enter__(self):
        """ Entering a with statement """
        return self


    def __exit__(self, _type, value, traceback):
        """ Exit a with statement"""
        if traceback is not None:
            # Exception occurred:
            logging.error('Exiting with statement with exception %s' % traceback)

        self.cleanup()


    #Reset hardware on robot when killed
    def cleanup(self):
        """ Reset the hardware"""
        logging.info('Stopping the car, resetting hardware.')
        self.back_wheels.speed = 0
        self.front_wheels.turn(90)
        self.camera.release()
        self.video_lane.release()
        cv2.destroyAllWindows()


    #Drive method
    def drive(self, speed=__INITIAL_SPEED):
        """ Main entry point of the car, and put it in drive mode
        Keyword arguments:
        speed -- speed of back wheel, range is 0 (stop) - 100 (fastest)
        """
        logging.info('Starting to drive at speed %s...' % speed)
        
        #Speed of the car
        self.back_wheels.speed = speed
       
        #Live camera processing
        while self.camera.isOpened():
            _, image_lane = self.camera.read()

            image_lane = self.follow_lane(image_lane)
            image_lane=cv2.resize(image_lane,(340,240))#Resize for codex + corruption
            self.video_lane.write(image_lane)
            
            show_image('Lane Lines', image_lane)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cleanup()
                break


    def follow_lane(self, image):
        image = self.lane_follower.follow_lane(image)
        return image



############################
# Utility Functions
############################
#Show image (window creation)
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)




############################
# MAIN
############################
def main():
    with PyLot() as car:
        car.drive(40)

#Startup
if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    main()