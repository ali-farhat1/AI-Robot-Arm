import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
import pybullet as p
import pybullet_data
import random


# Some Varibles From Before

URDF_PATH = "/home/ali/ros2_arm/src/robot_control/robot_control/urdf/Robot_stable.urdf"

# The current active joints
ACTIVE_JOINTS = [9]

# The End-Effector number
END_EFFECTOR_INDEX = 17

# How high and low it should go, plus-minus the current number
HIGH_LOW_NOD = 0.15

# The Error number, since pybullet isnt always accurate
ERROR_ACCEPTANCE = 0.02

# How many steps per simulation
TOTAL_STEPS = 1000

# how many steps in the whole yk trainig, MAX_STEPS/TOTAL_STEPS = total episodes
MAX_STEPS = 100000

# Number of inputs and outputs
INPUTS = 7
OUTPUTS = 1

# Total numbers of nod per episode
NOD_PER_EPISODE = 3

# Start Pose:
start_pose = {
    1: np.deg2rad(90 - 90),  # 0
    3: np.deg2rad(90 - 90),  # 0
    6: np.deg2rad(45 - 90),  # -45 deg
    9: np.deg2rad(90 - 90),  # 0
    12: np.deg2rad(90 - 90), # 0
}

# Registering the ENV in the gymnsaioum enviournment, so we can just use our env with gym.make().abs
register(
    id = "robotarm-env",                         # Name of the gymnsaioum enviournment - I chose it
    entry_point= "RobotEnv:RobotArmEnv"          # FileName:ClassName
)



class RobotArmEnv(gym.Env):
    def __init__(self, render_mode=None):
        
        # Basically the out put by the model, which will be ONE number
        # Hence the shape, and it will be from -1.00 to 1.00, for the 
        self.action_space = gym.spaces.Box(
            low=-1.00, high=1.00, shape=(OUTPUTS,), dtype=np.float32
        )


        # 5 observations:
        #   [joint4_angle,
        #    tip_z, tip_z_velocity,
        #    phase,
        #    energy,
        #    nod_low,
        #    nod_high]  
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(INPUTS,), dtype=np.float32
        )

        # If I want it to seen by the Buetifal human EYE or no.
        self.render_mode = render_mode

        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # To get the BEUTIFAL plane ground from the libarbary which is already made.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    # This is for each start of an episode 
    def reset(self, seed=None):
        """Sets the base critery for the start of a new episode."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)
        self.energy = random.uniform(0.0, 1.0)
        
        # Set the base position.
        for joint_idx, angle in start_pose.items():
            p.resetJointState(self.robot, joint_idx, angle)



        start_pos = self._get_tip_z()
        self.nod_low  = start_pos + HIGH_LOW_NOD - ERROR_ACCEPTANCE
        self.nod_high  = start_pos - HIGH_LOW_NOD + ERROR_ACCEPTANCE

        self.step_count      = 0
        self.phase           = 0          # 0 = going up, 1 = going down
        self.prev_tip_z      = self._get_tip_z()
        self.prev_velocity   = 0.0
        self.jerk_accumulator = 0.0       # builds up jerkiness during a phase
        self.nods = 0                     # Total number of nods done

        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.prev_angle = np.rad2deg(float(states[0][0])) # The angle of previous tick
        self.angle = 0                   # Current Angle
        self.ticks_stopped = 0


        return self._get_obs(), {}

    
    def step(self, action):


        #scaled_action = action * 1.0 * self.energy

        target_angle = (action + 1) * 90 
    
        # Clamp to ensure it stays within physical limits
        target_angle = np.clip(target_angle, 0, 180)
        target_rad = np.deg2rad(target_angle)

        speed = 0.2 + self.energy * 0.8

        # Apply velocities to joint4
        p.setJointMotorControlArray(
            self.robot,
            ACTIVE_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=[target_rad], # Must be a list
            positionGains=[speed]
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
 
        # Advance the simulation by one tick
        p.stepSimulation()
        self.step_count += 1
 
        # ── Measure what happened ──────────────────────
        tip_z         = self._get_tip_z()
        tip_z_vel     = float(p.getJointState(self.robot, 9)[1])     # positive = moving up
        dt = 1/240
        acceleration  = (tip_z_vel - self.prev_velocity) / dt  # change in velocity = jerk signal
        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.angle = np.rad2deg(float(states[0][0]))
        angle_difference  = abs(self.angle - self.prev_angle)

        # Reward
        reward, truncated = self._reward(tip_z, tip_z_vel, angle_difference)
 
        # ── Update memory for next step ────────────────
        self.prev_tip_z    = tip_z
        self.prev_velocity = tip_z_vel
        self.prev_angle = self.angle
 
        # ── Episode end ────────────────────────────────
        terminated = False                         # nod never truly "ends"
        truncated = truncated or (self.step_count >= TOTAL_STEPS)

        return self._get_obs(), reward, terminated, truncated, {}



    # INTERNAL HELPERS    
    def _get_tip_z(self):
        """Return the Z height of the end effector (tool_link)."""
        link_state = p.getLinkState(self.robot, END_EFFECTOR_INDEX)
        return link_state[0][2]   # [0] = world position, [2] = Z component

    def _get_obs(self):
        """Build the observation vector safely."""
        joint_states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        # Extract positions
        joint_pos = [s[0] for s in joint_states]
        
        tip_z = self._get_tip_z()
        # Ensure velocity is captured correctly
        tip_z_vel = float(p.getLinkState(self.robot, END_EFFECTOR_INDEX, computeLinkVelocity=1)[7][2])

        return np.array(
            joint_pos + [tip_z, tip_z_vel, float(self.phase), self.energy, self.nod_low, self.nod_high],
            dtype=np.float32
        )

    
    def _reward(self, tip_z, tip_z_vel, angle_difference):
        reward = 0.0

        # 1. Continuous shaping: closer to current target = better, every single tick
        target_z = self.nod_high if self.phase == 0 else self.nod_low
        dist = abs(tip_z - target_z)
        reward += -dist  # simple negative distance; scale/tune as needed

        # 2. Small control cost so it doesn't slam the joint at max speed needlessly
        reward += -0.01 * angle_difference**2

        # 3. Phase completion bonus (sparse, but now supplements dense signal instead of being the only signal)
        phase_complete = False
        if self.phase == 0 and tip_z <= self.nod_high:
            phase_complete = True
            self.phase = 1
            reward += 5.0
        elif self.phase == 1 and tip_z >= self.nod_low:
            phase_complete = True
            self.phase = 0
            self.nods += 1
            reward += 5.0

        
        # 4. The amount of ticks it has stayed stopped, if more 
        if angle_difference == 0:
            self.ticks_stopped += 1
        else:
            self.ticks_stopped = 0

        if self.ticks_stopped > 0:
            reward -= min(0.1 * (1.15 ** self.ticks_stopped), 20.0) # grows every tick it stays frozen, no waiting for a threshold

        # 5. Full-episode success bonus
        truncated = False
        if self.nods >= NOD_PER_EPISODE:
            reward += 20.0
            truncated = True

        return reward, truncated