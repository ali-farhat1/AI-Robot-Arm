import serial
import ikpy.chain
import ikpy.utils.plot as plot_utils
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32MultiArray, String
import threading
from dataclasses import dataclass, field
import math
import numpy as np
from pathlib import Path

# Connect to Serial
print("OKAY")
#ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

# -----------------------------
# Load URDF
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # robot_control/robot_control

URDF_PATH = BASE_DIR / "urdf" / "Robot_stable.urdf"

my_chain = ikpy.chain.Chain.from_urdf_file(
    URDF_PATH,
    active_links_mask=[
        False, False, True,  # 0:Base, 1:fixed, 2:joint1
        False, True,         # 3:fixed, 4:joint2
        False, False, True,  # 5:fixed, 6:fixed, 7:joint3
        False, False, True,  # 8:fixed, 9:fixed, 10:joint4
        False, False, True,  # 11:fixed, 12:fixed, 13:joint5
        False, False, False, False
    ],
)

# Organizing the dataclasses
@dataclass
class Config:
    ik_start_pose: list = field(default_factory=lambda: [90, 90, 45, 90, 90, 10])
    actions: list = field(default_factory=lambda: ["up_down", "right_left"])
    action_load_time: float = 2

cfg = Config()
print("connected")

class RobotArm(Node):
    def __init__(self):
        super().__init__('RobotArm_Node')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.servo_publisher = self.create_publisher(Float32MultiArray, 'servo_angles', 10)
        self.ai_action = self.create_subscription(String, "/actions", self.action_callback, 10)
        self.live_servo_movements = self.create_subscription(
            Float32MultiArray, 'live_servo_movements', self.servo_movements_callback, 10)
        
        self.cur_x = None
        self.cur_y = None
        self.cur_z = None
        
        # Internally updates the angles of each joint
        self.s0 = None
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.s4 = None
        self.s5 = None
        
        self.starting_pose()
        
        # For Rviz
        self.rviz_timer = self.create_timer(0.2, self.update)
        # For Actions
        self.timer = self.create_timer(0.2, self.action_update)
        self.state = "idle"
        self.target = None
        self.original = None

        # Debugging purposes
        #self.manual_timer = self.create_timer(0.2, self.manual_command_loop)

    def starting_pose(self):
        self.s0, self.s1, self.s2, self.s3, self.s4, self.s5 = cfg.ik_start_pose
        self.sendCommand("N", self.s0, self.s1, self.s2, self.s3, self.s4, self.s5)
        print("Ready")

    def servo_movements_callback(self, msg):
        if len(msg.data) >= 6:
            self.s0, self.s1, self.s2, self.s3, self.s4, self.s5 = msg.data[:6]
        else:
            self.get_logger().warn("Received servo message with fewer than 6 elements!")

    def action_callback(self, msg):
        if msg.data not in cfg.actions:
            return
            
        if msg.data == "up_down":
            self.original = self.s3
            self.target = max(self.s3 - 40, 0)
            self.state = "move_to_target"
            self.sendCommand(mode='M', s0=self.s0, s1=self.s1, s2=self.s2, s3=self.target, s4=self.s4, s5=self.s5)
            
        elif msg.data == "right_left":
            self.original = self.s0
            self.target = min(self.s0 - 40, 180)
            self.state = "move_to_target"
            self.sendCommand(mode='M', s0=self.target, s1=self.s1, s2=self.s2, s3=self.s3, s4=self.s4, s5=self.s5)

    def action_update(self):
        # STEP 1: reached target → go back
        if self.state == "move_to_target":
            if abs(self.s3 - self.target) < 2 or abs(self.s0 - self.target) < 2:
                self.state = "move_to_home"
                self.sendCommand(
                    mode='M', 
                    s0=self.original + 40 if self.target == self.s0 else self.s0, 
                    s1=self.s1, 
                    s2=self.s2, 
                    s3=self.original + 40 if self.target == self.s3 else self.s3, 
                    s4=self.s4, 
                    s5=self.s5
                )
        # STEP 2: reached home → finish
        elif self.state == "move_to_home":
            if self.s3 == self.original or self.s0 == self.original:
                self.state = "idle"
                self.sendCommand(
                    mode='M', 
                    s0=self.original if self.target == self.s0 else self.s0, 
                    s1=self.s1, 
                    s2=self.s2, 
                    s3=self.original if self.target == self.s3 else self.s3, 
                    s4=self.s4, 
                    s5=self.s5
                )

    # ----- SENDING THE ANGLES TO RVIZ -----
    def update(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        msg.position = [
            math.radians(float(self.s0 - 90)),
            math.radians(float(self.s1 - 90)),
            math.radians(float(self.s2 - 90)),
            math.radians(float(self.s3 - 90)),
            math.radians(float(self.s4 - 90))
        ]
        self.publisher.publish(msg)

    def clamp_angle(self, a):
        return max(0, min(180, a))

    def sendCommand(self, mode, s0, s1, s2, s3, s4, s5=0, move_time=1):
        msg = Float32MultiArray()
        self.s0, self.s1, self.s2, self.s3, self.s4, self.s5 = s0, s1, s2, s3, 90, s5
        command = [float(s0), float(s1), float(s2), float(s3), float(s4), float(s5)]
        msg.data = command
        print(f"SENDING: {msg.data}")
        self.servo_publisher.publish(msg)
        self.update()

    def clamp_target(self, pos):
        x, y, z = pos
        limit = 0.4
        x = max(-limit, min(limit, x))
        y = max(-limit, min(limit, y))
        z = max(0.02, min(0.4, z))
        return [x, y, z]

    def forwardKine(self):
        full_ik_radians = np.zeros(len(my_chain.links))
        full_ik_radians[2] = math.radians(self.s0 - 90)
        full_ik_radians[4] = math.radians(self.s1 - 90)
        full_ik_radians[7] = math.radians(self.s2 - 90)
        full_ik_radians[10] = math.radians(self.s3 - 90)
        full_ik_radians[13] = math.radians(self.s4 - 90)
        
        transformation_matrix = my_chain.forward_kinematics(full_ik_radians)
        return transformation_matrix[0, 3], transformation_matrix[1, 3], transformation_matrix[2, 3]

    def move_smooth(self, x, y, z, claw_angle=0, step_deg=6, mode='M'):
        current_angles = np.zeros(len(my_chain.links), dtype=float)
        current_angles[2] = math.radians((self.s0 or 90) - 90)
        current_angles[4] = math.radians((self.s1 or 90) - 90)
        current_angles[7] = math.radians((self.s2 or 90) - 90)
        current_angles[10] = math.radians((self.s3 or 90) - 90)
        current_angles[13] = math.radians((self.s4 or 90) - 90)
        
        clamped_pos = self.clamp_target([x, y, z])
        target = my_chain.inverse_kinematics(clamped_pos, initial_position=current_angles)
        
        if target is None:
            return False, False
            
        self.s0 = math.degrees(target[2]) + 90
        self.s1 = math.degrees(target[4]) + 90
        self.s2 = math.degrees(target[7]) + 90
        self.s3 = math.degrees(target[10]) + 90
        self.s4 = math.degrees(target[13]) + 90
        self.s5 = claw_angle
        
        return self.sendCommand(mode='M', s0=self.s0, s1=self.s1, s2=self.s2, s3=self.s3, s4=90, s5=self.s5, move_time=step_deg)

    def drawcircle(self, h, k, r, z=0.1, points=50, move_time=0.2):
        for theta in np.linspace(0, 2 * math.pi, points):
            x = h + r * math.cos(theta)
            y = k + r * math.sin(theta)
            self.move_smooth(x=x, y=y, z=z, step_deg=move_time)

    def manual_command_loop(self):
        fk = self.forwardKine()
        print(f"FK Result: {fk}")
        print("\n--- Enter Target Coordinates ---")
        try:
            x = float(input("X: "))
            y = float(input("Y: "))
            z = float(input("Z: "))
            self.move_smooth(x=x, y=y, z=z, step_deg=2.0, mode='centering')
            print(f"FK Result: {self.forwardKine()}")
        except Exception as e:
            print(f"ERROR: {e}")

def main():
    rclpy.init()
    node = RobotArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()