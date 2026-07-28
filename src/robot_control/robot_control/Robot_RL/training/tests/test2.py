import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
import pybullet as p
import pybullet_data
import random


# Some Varibles From Before

URDF_PATH = "/home/ali/ros2_arm/src/my_robot_scripts/my_robot_scripts/Robot_stable.urdf"

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
INPUTS = 1
OUTPUTS = 5

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
            low=-1.00, high=1.00, shape=(INPUTS,), dtype=np.float32
        )


        # 5 observations:
        #   [joint4_angle,
        #    tip_z, tip_z_velocity,
        #    phase,
        #    energy]  
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OUTPUTS,), dtype=np.float32
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


        return self._get_obs(), {}

    
    def step(self, action):


        scaled_action = action * 1.0 * self.energy

        target_angle = (action + 1) * 90 
    
        # 2. Clamp to ensure it stays within physical limits
        target_angle = np.clip(target_angle, 0, 180) * self.energy

        # Apply velocities to joint4
        p.setJointMotorControlArray(
            self.robot,
            ACTIVE_JOINTS,
            p.POSITION_CONTROL,
            targetPositions=target_angle
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
        acceleration  = tip_z_vel - self.prev_velocity  # change in velocity = jerk signal
 
        # Reward
        reward, truncated = self._reward(tip_z_vel, tip_z, acceleration)
 
        # ── Update memory for next step ────────────────
        self.prev_tip_z    = tip_z
        self.prev_velocity = tip_z_vel
 
        # ── Episode end ────────────────────────────────
        terminated = False                         # nod never truly "ends"
        #truncated  = self.step_count >= TOTAL_STEPS or self.nods == NOD_PER_EPISODE # just reset after enough steps or 3 nods.
        truncated  = self.step_count >= TOTAL_STEPS

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
            joint_pos + [tip_z, tip_z_vel, float(self.phase), self.energy],
            dtype=np.float32
        )

    
    def _reward(self, tip_z_vel, tip_z, acceleration):
            # ── Reward calculation ─────────────────────────
            reward = 0.0
            truncated = False

            # If it is even moving:
            if tip_z_vel == 0:
                reward-=1
            else:
                reward+=1 
    
            # Velocity reward: are we moving in the correct direction for this phase?
            if self.phase == 0:   # should be going UP → negative velocity is good
                if tip_z_vel < 0:
                    # if it is going in the right direction, if the veolocity is right
                    reward += max(0.0, -tip_z_vel) * 10.0
                else:
                    # if it is going in the wrong direction
                    reward -= 10
            else:                 # should be going DOWN → positive velocity is good
                if tip_z_vel > 0:
                    # if it is going in the right direction, if the veolocity is right
                    reward += max(0.0, tip_z_vel) * 10.0
                else:
                    # if it is going in the wrong direction
                    reward -= 10

    
            # Jerkiness penalty: sudden changes in velocity are bad
            jerk = abs(acceleration)
            reward -= jerk * 5.0


            # Accumulate jerk over the whole phase (used for bonus calculation)
            self.jerk_accumulator += jerk
    
            if self.phase == 0:
                if tip_z > self.nod_high:
                    reward -= 10 * (tip_z - self.nod_high)
            else:
                if tip_z < self.nod_low:
                    reward -= 30 * (self.nod_low - tip_z)
            

            #  Phase completion: did we reach the target height?
            phase_complete = False
            #print("tip_z:", tip_z, "vel:", tip_z_vel, "nod_high:", self.nod_high)
            if self.phase == 0 and tip_z <= self.nod_high:
                phase_complete = True
            elif self.phase == 1 and tip_z >= self.nod_low:
                phase_complete = True
    
            if phase_complete:
                # Quality bonus: 50 if perfectly smooth, less if jerky
                # We normalise jerk_accumulator so small jerk → bonus close to 50
                smoothness = 1.0 / (1.0 + self.jerk_accumulator)  # between 0 and 1
                reward += 50.0 * smoothness
    
                # Flip phase and reset jerk tracker for next phase
                self.phase = 1 - self.phase   # 0→1 or 1→0
                self.jerk_accumulator = 0.0

                # If the full 0-1 phase is done then add
                if self.phase == 0:
                    self.nods += 1
                
                # If it completed the number of nods per episode
                if self.nods == NOD_PER_EPISODE:
                    reward += 1000
                    truncated = True

            return reward, truncated