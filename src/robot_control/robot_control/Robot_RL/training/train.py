import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from robot_control.Robot_RL.environments.robot_env import RobotArmEnv
import time

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():

    train_env = Monitor(RobotArmEnv(render_mode=None))
    eval_env  = Monitor(RobotArmEnv(render_mode=None))

    model = sb3_class(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="auto",
        tensorboard_log="logs/",
        ent_coef = 0.05
    )

    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=300,
        verbose=1
    )

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
        verbose=1
    )

    best_model_save_path = os.path.join(model_dir, f"{args.sb3_algo}_best_model")

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_save_path,
        verbose=1
    )
    
    """
    total_timesteps: pass in a very large number to train (almost) indefinitely.
    tb_log_name: create log files with the name [gym env name]_[sb3 algorithm] i.e. Pendulum_v1_SAC
    callback: pass in reference to a callback fuction above
    """
    model.learn(total_timesteps=int(1e10), tb_log_name=f"{args.gymenv}_{args.sb3_algo}", callback=eval_callback)

def test():        
    model = sb3_class.load(os.path.join(model_dir, f"{args.sb3_algo}_best_model", "best_model"), env=env)
    #model = sb3_class.load("/home/ali/ros2_arm/src/my_robot_scripts/my_robot_scripts/models/A2C_best_model/best_model.zip", env=env)
    obs = env.reset()[0] 
    
    env.energy = energy

    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        time.sleep(1./100.)
        if terminated or truncated:
            #break
            pass


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('energy', nargs='?', help="How much energy")    
    parser.add_argument('--test', help='Test mode', action='store_true')

    args = parser.parse_args()

    # Dynamic way to import algorithm. For example, passing in DQN is equivalent to hardcoding:
    # from stable_baselines3 import DQN
    sb3_class = getattr(stable_baselines3, args.sb3_algo)

    if args.test:
        #env = gym.make(args.gymenv, render_mode='human')
        env = RobotArmEnv(render_mode="human")
        try:
            energy = float(args.energy)
        except Exception as e:
            print("You have to add another number in float for energy")

        test()
    else:
        #env = gym.make(args.gymenv)
        env = RobotArmEnv()
        env = Monitor(env)
        # env = gym.wrappers.RecordVideo(env, video_folder=recording_dir, episode_trigger = lambda x: x % 10000 == 0)
        train()
        
