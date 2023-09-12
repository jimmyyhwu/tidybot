# tidybot

This code release accompanies the following project:

### TidyBot: Personalized Robot Assistance with Large Language Models

Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, Thomas Funkhouser

*Autonomous Robots (AuRo) - Special Issue: Large Language Models in Robotics*, 2023

*IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2023

[Project Page](https://tidybot.cs.princeton.edu) | [PDF](https://tidybot.cs.princeton.edu/paper.pdf) | [arXiv](https://arxiv.org/abs/2305.05658) | [Video](https://youtu.be/bCkDynX1KmQ)

**Abstract:** For a robot to personalize physical assistance effectively, it must learn user preferences that can be generally reapplied to future scenarios. In this work, we investigate personalization of household cleanup with robots that can tidy up rooms by picking up objects and putting them away. A key challenge is determining the proper place to put each object, as people's preferences can vary greatly depending on personal taste or cultural background. For instance, one person may prefer storing shirts in the drawer, while another may prefer them on the shelf. We aim to build systems that can learn such preferences from just a handful of examples via prior interactions with a particular person. We show that robots can combine language-based planning and perception with the few-shot summarization capabilities of large language models (LLMs) to infer generalized user preferences that are broadly applicable to future interactions. This approach enables fast adaptation and achieves 91.2% accuracy on unseen objects in our benchmark dataset. We also demonstrate our approach on a real-world mobile manipulator called TidyBot, which successfully puts away 85.0% of objects in real-world test scenarios.

![](https://github.com/jimmyyhwu/tidybot/assets/6546428/dc1e87d9-f931-4741-916c-3dd11127b485) | ![](https://github.com/jimmyyhwu/tidybot/assets/6546428/6b21b3c3-d360-4d58-89d6-6c1dfd57ac19) | ![](https://github.com/jimmyyhwu/tidybot/assets/6546428/73f61e30-90ad-4f40-a5c8-1007adad897b)
:---: | :---: | :---:
![](https://github.com/jimmyyhwu/tidybot/assets/6546428/643fe718-12b2-44ca-afa7-f6ff2ddc304c) | ![](https://github.com/jimmyyhwu/tidybot/assets/6546428/afc749b6-0c3b-4b8e-a0d2-ddd1ba0a6bf5) | ![](https://github.com/jimmyyhwu/tidybot/assets/6546428/12e51dea-34a4-4c91-8423-34757bb485b7)

## Overview

Here is an overview of how this codebase is organized:

* [`server`](server): Server code for TidyBot (runs on GPU workstation)
* [`robot`](robot): Robot code for TidyBot (runs on mobile base computer)
* [`stl`](stl): Files for 3D printed parts
* [`benchmark`](benchmark): Code for the benchmark dataset

## Setup

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) environments. Our setup (tested on Ubuntu 20.04.6 LTS) uses the following 3 environments:

1. `tidybot` env on the server for general use
2. `tidybot` env on the robot for general use
3. `vild` env on the server for object detection only

See the respective READMEs inside the [`server`](server) and [`robot`](robot) directories for detailed setup instructions.

## TidyBot Quickstart

Unless otherwise specified, the `tidybot` Conda env should always be used:

```bash
conda activate tidybot
```

### Teleoperation Mode

We provide a teleoperation interface ([`teleop.py`](server/teleop.py)) to operate the robot using primitives such as pick, place, or toss.

First, run this command to start the teleop interface on the server (workstation), where `<robot-num>` is `1`, `2`, or `3`, depending on the robot to be controlled:

```bash
python teleop.py --robot-num <robot-num>
```

On the robot (mobile base computer), make sure that the convenience stop and mobile base driver are both running. Then, run this command to start the controller:

```bash
python controller.py
```

Once the server and robot both show that they have successfully connected to each other, use these controls to teleop the robot:

* Click on the overhead image to select waypoints
* Press `<Enter>` to execute selected waypoints on the robot
* Press `<Esc>` to clear selected waypoints or to stop the robot
* Press `0` through `5` to change the selected primitive
* Press `q` to quit
* If necessary, use the convenience stop to kill the controller
* If necessary, use the e-stop to cut power to the robot (the mobile base computer will stay on)

Notes:

* If keypresses are not registering, make sure that the teleop interface is the active window
* The default primitive (index `0`) is movement-only (no arm). To use the arm, you will need to change the selected primitive to something else. Check the terminal output to see the list of all primitives as well as the currently selected primitive.

---

To generate paths with an occupancy map rather than manually clicking waypoints, use the `--shortest-path` flag.

```bash
python teleop.py --robot-num <robot-num> --shortest-path
```

This will load the receptacles specified in [`scenarios/test.yml`](server/scenarios/test.yml) as obstacles and build an occupancy map to avoid running into them.

---

For additional debugging visualization, the `--debug` flag can be used.

Server:

```bash
python teleop.py --robot-num <robot-num> --debug
```

Robot:

```bash
python controller.py --debug
```

### Fully Autonomous Mode

To operate the robot in fully autonomous mode, we use the demo interface in [`demo.py`](server/demo.py). By default, the demo will load the test scenario in [`scenarios/test.yml`](server/scenarios/test.yml) along with the corresponding LLM-summarized user preferences in [`preferences/test.yml`](server/preferences/test.yml).

To start the demo on the server, first start the object detector server with the `vild` Conda env:

```bash
conda activate vild
python object_detector_server.py
```

Then, in a separate terminal, start the demo interface (with the `tidybot` env):

```bash
python demo.py --robot-num <robot-num>
```

On the robot, make sure that the convenience stop and mobile base driver are both running. Then, run this command to start the controller:

```bash
python controller.py
```

These are the controls used to run the demo:

* Press `<Enter>` to start the robot
* Press `<Esc>` to stop the robot at any time
* Press `0` to enter supervised mode (the default mode), in which the robot will wait for human approval (via an `<Enter>` keypress) before executing every command
* Press `1` to enter autonomous mode, in which the robot will start executing commands whenever `<Enter>` is pressed and stop moving whenever `<Esc>` is pressed
* Press `q` to quit
* If necessary, use the convenience stop to kill the controller
* If necessary, use the e-stop to cut power to the robot (the mobile base computer will stay on)

Note: If keypresses are not registering, make sure that the demo interface is the active window.

---

To load a different scenario (default is `test`), use the `--scenario-name` argument:

```bash
python demo.py --robot-num <robot-num> --scenario-name <scenario-name>
```

For example, to load `scenario-08` and use robot #1, you can run:

```bash
python demo.py --robot-num 1 --scenario-name scenario-08
```

---

For additional debugging visualization, the `--debug` flag can be used.

Server:

```bash
python demo.py --robot-num <robot-num> --debug
```

Robot:

```bash
python controller.py --debug
```

## Troubleshooting

### Mobile Base Accuracy

The marker detection setup should output 2D robot pose estimates with centimeter-level accuracy. For instance, our setup can reliably pick up small Lego Duplo blocks (32 mm x 32 mm) from the floor. Inaccurate marker detection can be due to many reasons, such as inaccurate camera alignment or suboptimal camera settings (e.g., exposure and gain, see `get_video_cap` in [`utils.py`](server/utils.py)). Also note that the mobile base motors should be calibrated (`.motor_cal.txt`) for more accurate movement.

### Arm Accuracy

The 3 Kinova arms are repeatable but have slightly different zero heading positions, so they require some compensation to be consistent with each other. See the arm-dependent heading compensation in [`controller.py`](robot/controller.py).

### Server Ports

If multiple people have been using the server, you may run into this error:

```
OSError: [Errno 98] Address already in use
```

To kill all processes using the occupied ports, you can use the [`clear-ports.sh`](server/clear-ports.sh) script (requires sudo):

```bash
./clear-ports.sh
```

For reference, here are all of the ports used by this codebase:
* `6000`: Camera server (top camera)
* `6001`: Camera server (bottom camera)
* `6002`: Marker detector server
* `6003`: Object detector server
* `6004`: Robot 1 controller server
* `6005`: Robot 2 controller server
* `6006`: Robot 3 controller server
* `6007`: Robot 1 control
* `6008`: Robot 2 control
* `6009`: Robot 3 control
* `6010`: Robot 1 camera
* `6011`: Robot 2 camera
* `6012`: Robot 3 camera

### Camera Errors

The overhead cameras may occasionally output errors such as this:

```
[ WARN:16@1367.080] global /io/opencv/modules/videoio/src/cap_v4l.cpp (1013) tryIoctl VIDEOIO(V4L2:/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_E4298F4E-video-index0): select() timeout.
[ WARN:16@2049.229] global /io/opencv/modules/videoio/src/cap_v4l.cpp (1013) tryIoctl VIDEOIO(V4L2:/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_099A11EE-video-index0): select() timeout.
Corrupt JPEG data: 36 extraneous bytes before marker 0xd9
Corrupt JPEG data: premature end of data segment
```

Typically, these errors can be resolved by unplugging the camera and plugging it back in.

Be sure to also check the quality and length of the USB extension cable, as USB 2.0 does not support cable lengths longer than 5 meters.

## Citation

If you find this work useful for your research, please consider citing:

```
@article{wu2023tidybot,
  title = {TidyBot: Personalized Robot Assistance with Large Language Models},
  author = {Wu, Jimmy and Antonova, Rika and Kan, Adam and Lepert, Marion and Zeng, Andy and Song, Shuran and Bohg, Jeannette and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  journal = {Autonomous Robots},
  year = {2023}
}
```
