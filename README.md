### Dependencies
* OpenCV 4.2
* Eigen 3
* Ceres 2.0.0
* G2O (tag:20230223_git)
* TensorRT 8.6.1.6
* CUDA 12.1
* python
* ROS noetic
* yaml 0.5.3
* Boost
* yolov8n


dataroot
├── cam0
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
├── cam1
│   └── data
│       ├── t0.jpg
│       ├── t1.jpg
│       ├── t2.jpg
│       └── ......
└── imu0
    └── data.csv

## :computer: Build
```
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    catkin_init_workspace
    cd ..
    catkin_make
    cd ~/catkin_ws/src
    git clone https://github.com/1910317615/PLCDSLAM/tree/master
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

