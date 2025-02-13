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
            'motors = attempt1.motors:main',
            'sonar = attempt1.sonar:main',
            'control = attempt1.control:main',
            'camera = attempt1.camera:main',
            'locate = attempt1.locate:main',
            'move = attempt1.move:main',
            'save = attempt1.save:main'
        ],
    },
)
