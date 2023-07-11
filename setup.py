from setuptools import setup
import os
from glob import glob
package_name = 'hailo8_detection'
setup(
    name=package_name,
    version='0.0.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
        (os.path.join('share', package_name), glob('./models/*.hef')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel Tobias',
    maintainer_email='djtobias@aggies.ncat.edu',
    description='yolov8 node for Hailo8',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'hailo8_detection = hailo8_detection.hailo8_detection:main',
        ],
    },
)