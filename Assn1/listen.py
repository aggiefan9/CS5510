from pynput import keyboard
from pynput.keyboard import Key

from Car import Car
import time


def press_callback(key):
    try:
        # print('alphanumeric key {} pressed'.format(key.char))
        if key.char == 'w':
            print('moving forward')
            # car.control_car(100, 100)
        elif key.char == 'a':
            print('turning left')
            # car.control_car(-100, 100)
        elif key.char == 's':
            print('moving backward')
            # car.control_car(-100, -100)
        elif key.char == 'd':
            print('turning right')
            # car.control_car(100, -100)
        else:
            print("invalid action")
            # car.control_car(0, 0)
    except AttributeError:
        # print('special key {} pressed'.format(key))
        if key == Key.up:
            print('moving forward')
            # car.control_car(100, 100)
        elif key == Key.down:
            print('moving backward')
            # car.control_car(-100, -100)
        elif key == Key.right:
            print('turning right')
            # car.control_car(100, -100)
        elif key == Key.left:
            print('turning left')
            # car.control_car(-100, 100)
        elif key == Key.esc:
            print('quiting...')
        else:
            print("invalid action")
            # car.control_car(0, 0)

def release_callback(key):
    # print('{} released'.format(key))
    try:
        if key.char in ['w','a','s','d']:
            print('stopped moving/turning')
            # car.control_car(0, 0)
    except AttributeError:
        if key in [Key.up,Key.left,Key.down,Key.right]:
            print('stopped moving/turning')
            # car.control_car(0, 0)
    if key == Key.esc:
        return False

l = keyboard.Listener(on_press=press_callback, on_release=release_callback)
l.start()
l.join()