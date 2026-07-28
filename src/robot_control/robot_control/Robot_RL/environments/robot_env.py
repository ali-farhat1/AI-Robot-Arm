import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
import pybullet as p
import pybullet_data
import random


URDF_PATH = "/home/ali/ros2_arm/src/robot_control/robot_control/urdf/Robot_stable.urdf"

# Controlled joint
ACTIVE_JOINTS = [9]

# End-effector link index
END_EFFECTOR_INDEX = 17

# Target nod amplitude
HIGH_LOW_NOD = 0.15

# Position tolerance
ERROR_ACCEPTANCE = 0.02

# Maximum steps per episode
TOTAL_STEPS = 1000

# Total training steps
MAX_STEPS = 100000

# Number of previous actions included in the observation
ACTION_HISTORY_LENGTH = 10

# Observation and action dimensions
INPUTS = 7 + ACTION_HISTORY_LENGTH
OUTPUTS = 1

# Required nods to complete an episode
NOD_PER_EPISODE = 3

# Motion parameters
MAX_JOINT_SPEED_DEG_PER_SEC = 120.0
STILL_GRACE_TICKS = 50
JERK_WEIGHT = 0.005


# Initial robot pose
start_pose = {
    1: np.deg2rad(0),
    3: np.deg2rad(0),
    6: np.deg2rad(-45),
    9: np.deg2rad(0),
    12: np.deg2rad(0),
}


# Register the environment with Gymnasium
register(
    id="robotarm-env",
    entry_point="RobotEnv:RobotArmEnv"
)


class RobotArmEnv(gym.Env):
    def __init__(self, render_mode=None):

        # Action produced by the policy
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OUTPUTS,), dtype=np.float32
        )

        # Observation vector:
        # joint angle, tip height, tip velocity, phase,
        # energy, nod limits, and recent actions.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(INPUTS,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.action_history_length = ACTION_HISTORY_LENGTH

        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Load built-in PyBullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )

        self.energy = random.uniform(0.0, 1.0)

        # Reset the robot to its initial pose
        for joint_idx, angle in start_pose.items():
            p.resetJointState(self.robot, joint_idx, angle)

        start_pos = self._get_tip_z()

        self.nod_low = start_pos - HIGH_LOW_NOD + ERROR_ACCEPTANCE
        self.nod_high = start_pos + HIGH_LOW_NOD - ERROR_ACCEPTANCE

        self.step_count = 0
        self.phase = 0          # 0 = moving up, 1 = moving down
        self.prev_tip_z = self._get_tip_z()
        self.prev_velocity = 0.0
        self.jerk_accumulator = 0.0
        self.nods = 0

        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.prev_angle = np.rad2deg(float(states[0][0]))
        self.angle = self.prev_angle

        self.ticks_stopped = 0
        self.prev_dist = 0

        # Initialize action history
        self.action_history = [0.0] * self.action_history_length

        return self._get_obs(), {}


    def step(self, action):

        dt = 1 / 240

        # Convert policy output to a target joint angle
        target_angle = np.clip((float(action[0]) + 1) * 90, 0, 180)
        target_rad = np.deg2rad(target_angle)

        # Scale joint speed using the current energy level
        max_speed_deg_per_sec = MAX_JOINT_SPEED_DEG_PER_SEC * (0.2 + self.energy * 0.8)
        max_velocity_rad = np.deg2rad(max_speed_deg_per_sec)

        p.setJointMotorControl2(
            self.robot,
            ACTIVE_JOINTS[0],
            p.POSITION_CONTROL,
            targetPosition=target_rad,
            positionGain=0.3,
            velocityGain=1.0,
            force=500,
            maxVelocity=max_velocity_rad,
        )

        for j, angle in start_pose.items():
            if j != 9:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=500
                )

        # Advance the simulation
        p.stepSimulation()
        self.step_count += 1

        # Measure the current state
        tip_z = self._get_tip_z()
        tip_z_vel = float(
            p.getLinkState(self.robot, END_EFFECTOR_INDEX, computeLinkVelocity=1)[6][2]
        )
        tip_z_accel = (tip_z_vel - self.prev_velocity) / dt

        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.angle = np.rad2deg(float(states[0][0]))
        angle_difference = abs(self.angle - self.prev_angle)

        reward, truncated = self._reward(
            tip_z,
            tip_z_accel,
            angle_difference,
        )

        # Update state for the next step
        self.prev_tip_z = tip_z
        self.prev_velocity = tip_z_vel
        self.prev_angle = self.angle

        self.action_history.pop(0)
        self.action_history.append(float(action[0]))

        terminated = False
        truncated = truncated or (self.step_count >= TOTAL_STEPS)

        return self._get_obs(), reward, terminated, truncated, {}


    # Helper methods
    def _get_tip_z(self):
        """Return the end-effector height."""
        return p.getLinkState(self.robot, END_EFFECTOR_INDEX)[0][2]

    def _get_obs(self):
        """Build the observation vector."""
        joint_states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        joint_pos = [s[0] for s in joint_states]

        tip_z = self._get_tip_z()
        tip_z_vel = float(
            p.getLinkState(
                self.robot,
                END_EFFECTOR_INDEX,
                computeLinkVelocity=1
            )[6][2]
        )

        return np.array(
            joint_pos
            + [
                tip_z,
                tip_z_vel,
                float(self.phase),
                self.energy,
                self.nod_low,
                self.nod_high,
            ]
            + self.action_history,
            dtype=np.float32,
        )


    def _reward(self, tip_z, tip_z_accel, angle_difference):
        reward = 0.0

        # Reward progress toward the current target
        target_z = self.nod_high if self.phase == 0 else self.nod_low

        prev_dist = self.prev_dist
        dist = abs(tip_z - target_z)
        self.prev_dist = dist

        reward += (prev_dist - dist) * 2

        # Penalize jerky motion
        self.jerk_accumulator = (
            0.9 * self.jerk_accumulator
            + 0.1 * abs(tip_z_accel)
        )
        reward += -JERK_WEIGHT * self.jerk_accumulator

        # Penalize unnecessary joint movement
        reward += -0.01 * angle_difference ** 2

        # Reward reaching the current target
        if self.phase == 0 and tip_z >= self.nod_high:
            self.phase = 1
            self.prev_dist = 0
            reward += 30.0

        elif self.phase == 1 and tip_z <= self.nod_low:
            self.phase = 0
            self.nods += 1
            self.prev_dist = 0
            reward += 30.0

        # Penalize remaining still for too long
        if angle_difference < 1.0:
            self.ticks_stopped += 1
        else:
            self.ticks_stopped = 0

        if self.ticks_stopped > STILL_GRACE_TICKS:
            stuck_ticks = self.ticks_stopped - STILL_GRACE_TICKS
            reward -= min(0.05 * stuck_ticks, 5.0)

        # Bonus for completing all required nods
        truncated = False
        if self.nods >= NOD_PER_EPISODE:
            reward += 100.0
            truncated = True

        return reward, truncated
