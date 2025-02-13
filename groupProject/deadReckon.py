from Car import Car
import time

if __name__ == '__main__':
    car = Car()
    ## Constant values
    speed = 100
    drive_time = 2
    turn_time = 0.95
    pause_time = 0.1
    
    ## Navigate in a square
    for i in range(4):
        ## Move Forward
        car.control_car(speed, speed)
        time.sleep(drive_time)
        car.control_car(0, 0)
        time.sleep(pause_time)
        ## Turn Left
        car.control_car(-speed, speed)
        time.sleep(turn_time)
        car.control_car(0, 0)
        time.sleep(pause_time)
    
    ## Extra stop just in case
    car.control_car(0, 0)