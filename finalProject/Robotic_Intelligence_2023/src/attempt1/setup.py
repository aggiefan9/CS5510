from setuptools import find_packages, setup

package_name = 'attempt1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='pi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motors = attempt1.motors:main', ## Motor Control
            'sonar = attempt1.sonar:main', ## Sonar sensors
            'control = attempt1.control:main', ## Taylor Anderson's Object Distance Demo
            'camera = attempt1.camera:main', ## Takes pictures (not working)
            'locate = attempt1.locate:main', ## Detects the ball
            'move = attempt1.move:main', ## Moves the robot to the goal
            'goal = attempt1.locate_goal:main', ## Detects the goal
            'return = attempt1.return:main', ## Moves the robot to the goal
            'save = attempt1.save:main', ## Saves images (used for camera calibration)
            'map = attempt1.mapping:main', ## Occupancy grid (not working)
            'plan = attempt1.planning:main' ## Motion PLanner (not working)
        ],
    },
)
