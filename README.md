# AI Robot Arm

An AI-powered personal robot built using **ROS2**, combining robotic control, reinforcement learning, and large language models.

## Features

* 🤖 ROS2-based modular architecture
* 🦾 6-DOF robotic arm control
* 🧠 AI assistant powered by an LLM
* 📚 Memory system for conversations and knowledge
* 🎯 Reinforcement learning for robot behaviors
* 📡 Ultrasonic distance sensing
* 💻 Simple user interface for interacting with the robot

## Project Structure

```text
src/
├── robot_ai/             # AI brain and memory system
├── robot_control/        # Robot arm control and reinforcement learning
├── robot_ui/             # User interface
└── ultrasonic_mapping/   # Ultrasonic sensor integration (Maps ultrasonoic input to 3d Rviz)
```

## Technologies

* ROS2 Humble
* Python
* Reinforcement Learning (Stable-Baselines3)
* PyBullet
* OpenRouter API
* URDF
* RViz

## Installation

Clone the repository:

```bash
git clone https://github.com/noiceman6298/AI-Robot-Arm.git
```

Build the workspace:

```bash
cd AI-Robot-Arm
colcon build
source install/setup.bash
```

## Current Status

This project is actively under development. New features, improvements, and documentation are added regularly.

## Future Goals

* Computer vision integration
* Voice interaction
* Improved memory system
* Better reinforcement learning policies

