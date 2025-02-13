import numpy as np

def create_map():
    map = [['sec1.jpg', 'sec2.jpg', 'sec3.jpg', 'sec4.jpg'], ['sec5.jpg', 'sec6.jpg', 'sec7.jpg', 'sec8.jpg'], ['sec9.jpg', 'sec10.jpg', 'sec11.jpg', 'sec12.jpg'], ['sec13.jpg', 'sec14.jpg', 'sec15.jpg', 'sec16.jpg']]
    map[np.random.randint(0,4)][np.random.randint(0,4)] = 'AlanStringer_Hiker.jpg'
    print('Map Loaded')
    return map

if __name__ == "__main__":
    create_map()