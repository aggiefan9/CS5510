from Car import Car
from Photographer import Photographer
from drive import keyTracker
# from sshkeyboard import listen_keyboard_manual
import time
import math
import sys


max_fps = 20
## Constant values
speed = 100
drive_time = 2
turn_time = .78
pause_time = 0.1


class PoseTracker:
    ## Constant values
    speed = 0.3 #m/s
    turn_speed = 2 #rad/s
    
    #convention: [x_pos, y_pos, z_pos, yaw in rads]
    current_pose = [0, 0, 0, 0]
    last_update = time.time()
    
    def updatePose(self, drive_vector):
        delta_time = time.time() - self.last_update
        self.last_update = time.time()
        
        
        #update position and angle based on drive_vector
        self.current_pose[0] += self.speed * delta_time * math.cos(self.current_pose[3]) * drive_vector[0]
        self.current_pose[2] += self.speed * delta_time * math.sin(self.current_pose[3]) * drive_vector[0]
        #z_pos assumed to not change
        
        self.current_pose[3] += self.turn_speed * delta_time * drive_vector[1]
    
    def writePose(self):
        f = open("poses/001.txt", "a")
        f.write("x x x {} x x x {} x x x {}\n".format(self.current_pose[0], self.current_pose[1], self.current_pose[2]))
        f.close()
        
   
def trackingSleep(sleepTime, FPS, pt, pg, drive_vector):
    entered = time.time()
    last_frame = 0
    #assume it takes some amount of time to do the pose update and image writing
    while (time.time() - entered) < (sleepTime - 0.25):
        try:
            
            if (time.time() - last_frame) > (1.0 / max_fps):
                last_frame = time.time()
                pt.updatePose(drive_vector)
                print("Taking a picture!")
                pg.takePicture()
                pt.writePose()
                
                
        except KeyboardInterrupt:
            print("Exiting...")
            # Release the camera
            sys.exit()
    leftover = sleepTime - (time.time() - entered)
    if leftover > 0:
        time.sleep(leftover)
    
    
if __name__ == '__main__':

    pt = PoseTracker()
    pg = Photographer()
    

    car = Car()
    
    ## Navigate in a square
    for i in range(4):
        ## Move Forward
        car.control_car(speed, speed)
        trackingSleep(drive_time, max_fps, pt, pg, [1, 0])
        car.control_car(0, 0)
        time.sleep(pause_time)
        ## Turn Left
        car.control_car(-speed, speed)
        trackingSleep(turn_time, max_fps, pt, pg, [0, -1])
        car.control_car(0, 0)
        time.sleep(pause_time)
    
    ## Extra stop just in case
    car.control_car(0, 0)