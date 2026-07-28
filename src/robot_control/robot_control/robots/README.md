# Robot Arm ROS2 Setup Guide

This README explains how to connect the ESP32, start the micro-ROS agent, and run the ROS2 robot arm system.

---

## 1. Connect ESP32 USB

### Ubuntu (Native)

Connect the ESP32 and check the available USB ports:

```bash
ls /dev/ttyUSB*
```

The ESP32 should appear as:

```bash
/dev/ttyUSB0
```

The port may be different depending on the computer.

---

### Windows + WSL

If using Ubuntu through WSL, the ESP32 USB device must first be attached.

Open **Windows Terminal as Administrator**.

Check connected USB devices:

```bash
usbipd list
```

Find the ESP32 USB device and note its BUSID.

Example:

```
BUSID    DEVICE
2-8      USB Serial Device
```

Attach the device to WSL:

```bash
usbipd attach --wsl --busid 2-8
```

Then inside Ubuntu check the USB port:

```bash
ls /dev/ttyUSB*
```

The ESP32 should now appear:

```
/dev/ttyUSB0
```

---

# 2. Start micro-ROS Agent

The ESP32 communicates with ROS2 using the micro-ROS agent. The agent must be running before starting any ROS2 nodes.

Go to the micro-ROS workspace:

```bash
cd ~/uros_ws
```

Build the workspace:

```bash
colcon build
```

Source the workspace:

```bash
source install/setup.bash
```

Run the micro-ROS agent:

```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
```

Keep this terminal open while using the robot.

---

# 3. Start Robot ROS2 Nodes

Open a new terminal.

Source ROS2:

```bash
source /opt/ros/humble/setup.bash
```

Go to the robot workspace:

```bash
cd ~/ros2_arm
```

Build if needed:

```bash
colcon build
```

Source the workspace:

```bash
source install/setup.bash
```

Run the required ROS2 nodes:

```bash
ros2 run <package_name> <node_name>
```

---

# 4. Check ROS2 Connection

Check available topics:

```bash
ros2 topic list
```

You should see:

```
/servo_angles
/live_servo_movements
/ultrasonic_data
```

Check servo commands:

```bash
ros2 topic echo /servo_angles
```

Check live servo positions:

```bash
ros2 topic echo /live_servo_movements
```

---

# Troubleshooting

## micro_ros_agent package not found

If you get:

```
Package 'micro_ros_agent' not found
```

Run:

```bash
cd ~/uros_ws
colcon build
source install/setup.bash
```

Then try:

```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
```

---

## No executable found

Check that the agent executable exists:

```bash
ros2 pkg executables micro_ros_agent
```

Expected output:

```
micro_ros_agent micro_ros_agent
```

If it does not appear, rebuild:

```bash
cd ~/uros_ws
colcon build
source install/setup.bash
```

---

## USB Port Not Found

If `/dev/ttyUSB0` does not exist:

```bash
ls /dev/ttyUSB*
```

Check:

- ESP32 is connected
- WSL USB forwarding is enabled
- Correct USB port is being used

---

# Startup Order

Always start the robot in this order:

1. Connect ESP32 USB
2. Attach USB through WSL (if required)
3. Find the USB port:
   ```bash
   ls /dev/ttyUSB*
   ```
4. Start micro-ROS agent:
   ```bash
   cd ~/uros_ws
   source install/setup.bash
   ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200
   ```
5. Open a new terminal
6. Start ROS2 robot nodes

The robot arm is now connected and ready.