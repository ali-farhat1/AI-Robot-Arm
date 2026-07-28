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


# Action Histroy Lenght 
ACTION_HISTORY_LENGTH = 10

# Number of inputs and outputs
INPUTS = 7 + ACTION_HISTORY_LENGTH
OUTPUTS = 1

# Total numbers of nod per episode
NOD_PER_EPISODE = 3



# ---- NEW: real speed / smoothness controls -----------------------------
# Max degrees/second the joint is physically allowed to move. This is what
# actually stops the arm from reaching one extreme in a single tick -
# positionGain alone never capped speed, it's a PD gain, not a velocity
# limit. Tune this up if 3 nods can't fit inside TOTAL_STEPS, tune it down
# if motion still looks too snappy.
MAX_JOINT_SPEED_DEG_PER_SEC = 120.0

# How many consecutive "not really moving" ticks are OK before the
# stillness penalty kicks in. Gives the arm room to pause / reverse
# direction at the top or bottom of a nod without being punished for it.
STILL_GRACE_TICKS = 50

# Weight on the jerk (acceleration-change) penalty. Start small and
# increase if motion still looks jerky; decrease if the arm barely moves.
JERK_WEIGHT = 0.005
# --------------------------------------------------------------------------

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
        #    phase (0 going up, 1 down),
        #    energy,
        #    nod_low,
        #    nod_high, 
        #    action_history]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(INPUTS,), dtype=np.float32
        )

        # If I want it to seen by the Buetifal human EYE or no.
        self.render_mode = render_mode

        # The overall past actions it has
        self.action_history_length = ACTION_HISTORY_LENGTH

        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # To get the BEUTIFAL plane ground from the libarbary which is already made.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


    # This is for each start of an episode
    def reset(self, seed=None, options=None):
        """Sets the base critery for the start of a new episode."""
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)
        self.energy = random.uniform(0.0, 1.0)

        # Set the base position.
        for joint_idx, angle in start_pose.items():
            p.resetJointState(self.robot, joint_idx, angle)

        start_pos = self._get_tip_z()

        # FIXED: nod_low is now genuinely the LOWER target height and
        # nod_high the HIGHER one (they were swapped before - nod_low held
        # the +offset and nod_high held the -offset, which quietly flipped
        # the meaning of "phase 0 / going up" from the comment below).
        self.nod_low  = start_pos - HIGH_LOW_NOD + ERROR_ACCEPTANCE
        self.nod_high = start_pos + HIGH_LOW_NOD - ERROR_ACCEPTANCE

        self.step_count       = 0
        self.phase            = 0          # 0 = going up, 1 = going down
        self.prev_tip_z       = self._get_tip_z()
        self.prev_velocity    = 0.0
        self.jerk_accumulator = 0.0        # now actually used in _reward()
        self.nods             = 0          # Total number of nods done

        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.prev_angle    = np.rad2deg(float(states[0][0]))  # angle of previous tick
        self.angle         = self.prev_angle                  # FIXED: was hardcoded to 0
        self.ticks_stopped = 0
        self.prev_dist = 0

        # Becuase in the first run theirs no hisotry so it will crash
        self.action_history = [0.0] * self.action_history_length

        return self._get_obs(), {}


    def step(self, action):


        dt = 1 / 240

        # Absolute target angle from the policy's action - same mapping as before
        target_angle = np.clip((float(action[0]) + 1) * 90, 0, 180)
        target_rad = np.deg2rad(target_angle)

        # this is the actual speed control.
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

        # Advance the simulation by one tick
        p.stepSimulation()
        self.step_count += 1

        # ── Measure what happened ──────────────────────
        tip_z = self._get_tip_z()
        tip_z_vel = float(
            p.getLinkState(self.robot, END_EFFECTOR_INDEX, computeLinkVelocity=1)[6][2]
        )
        tip_z_accel = (tip_z_vel - self.prev_velocity) / dt

        states = p.getJointStates(self.robot, ACTIVE_JOINTS)
        self.angle = np.rad2deg(float(states[0][0]))
        angle_difference = abs(self.angle - self.prev_angle)

        # Reward
        reward, truncated = self._reward(tip_z, tip_z_accel, angle_difference)

        # ── Update memory for next step ────────────────
        self.prev_tip_z    = tip_z
        self.prev_velocity = tip_z_vel
        self.prev_angle    = self.angle
        self.action_history.pop(0) 
        self.action_history.append(float(action[0]))

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
        # FIXED: index [6] is linear velocity, not [7] (see step() note above)
        tip_z_vel = float(p.getLinkState(self.robot, END_EFFECTOR_INDEX, computeLinkVelocity=1)[6][2])

        return np.array(
            joint_pos + [tip_z, tip_z_vel, float(self.phase), self.energy,
                        self.nod_low, self.nod_high] + self.action_history,
            dtype=np.float32
        )


    def _reward(self, tip_z, tip_z_accel, angle_difference):
        reward = 0.0

        # 1. Continuous shaping: closer to current target = better, every single tick
        target_z = self.nod_high if self.phase == 0 else self.nod_low
        prev_dist = self.prev_dist
        dist = abs(tip_z - target_z)
        self.prev_dist = dist
        reward += (prev_dist - dist) * 2
        

        # 2. Smoothness penalty - NEW. jerk_accumulator and an
        # "acceleration" value both existed in the original code but
        # neither was ever actually plugged into the reward, so nothing
        # discouraged violent, snapping motion. This is that missing piece.
        self.jerk_accumulator = 0.9 * self.jerk_accumulator + 0.1 * abs(tip_z_accel)
        reward += -JERK_WEIGHT * self.jerk_accumulator

        # 3. Small control cost so it doesn't slam the joint at max speed needlessly
        reward += -0.01 * angle_difference ** 2

        # 4. Phase completion bonus (sparse, but now supplements dense signal instead of being the only signal)
        # FIXED comparisons to match the corrected nod_low/nod_high meaning
        if self.phase == 0 and tip_z >= self.nod_high:
            self.phase = 1
            self.prev_dist = 0
            reward += 30.0
        elif self.phase == 1 and tip_z <= self.nod_low:
            self.phase = 0
            self.nods += 1
            self.prev_dist = 0
            reward += 30.0

        # 5. The amount of ticks it has stayed stopped - FIXED to use a
        # grace period (STILL_GRACE_TICKS) and a linear, capped growth
        # instead of an immediate exponential one. The old version started
        # punishing on the very first still tick and hit its -20 cap
        # within ~38 ticks (0.16s of sim time) - bigger than the entire
        # episode-success bonus, for doing nothing more than pausing at
        # the top of a nod to reverse direction. That is what was forcing
        # the arm to slam back the instant it touched a target instead of
        # settling there.
        if angle_difference < 1.0:
            self.ticks_stopped += 1
        else:
            self.ticks_stopped = 0

        if self.ticks_stopped > STILL_GRACE_TICKS:
            stuck_ticks = self.ticks_stopped - STILL_GRACE_TICKS
            reward -= min(0.05 * stuck_ticks, 5.0)

        

        # 6. Full-episode success bonus
        truncated = False
        if self.nods >= NOD_PER_EPISODE:
            reward += 100.0
            truncated = True


        

        return reward, truncated



