from Car import Car
import time

if __name__ == '__main__':
    car = Car()
    
    ## Move Forward (Side 1)
    car.control_car(100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Turn Left
    car.control_car(-100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Move Forward (Side 2)
    car.control_car(100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Turn Left
    car.control_car(-100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Move Forward (Side 3)
    car.control_car(100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Turn Left
    car.control_car(-100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Move Forward (Side 4)
    car.control_car(100, 100)
    time.sleep(2)
    car.control_car(0, 0)
    time.sleep(0.1)
    ## Extra stop just in case
    car.control_car(0, 0)