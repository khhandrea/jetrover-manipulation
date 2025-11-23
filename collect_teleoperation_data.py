import os
import sys
import re
import copy
import pickle
import signal
import time
import numpy as np

from jetrover_gym.manipulation_env import JetRoverManipulationEnv

class PickleLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.step = 0

    def __call__(self, observation, action, reward, terminated=0, truncated=0, metadata=None):
        step = copy.deepcopy(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                metadata=metadata,
            )
        )
        self.data.append(step)
        self.step += 1

    def make_new_rollout(self, filename=None):
        if filename is not None:
            self.filename = filename
        self.data = []
        self.step = 0

    def save(self):
        print(f"Saving rollout to: {self.filename}")
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Done saving.")

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_help():
    print_yellow("  Teleop Controls:")
    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    m: to save demonstration")
    print_yellow("    h: help")
    print_yellow("    q: quit")
    print_yellow("  Belows can be typed at once e.g. 'aso(enter)'")
    print_yellow("    w, s : move upward/downward")
    print_yellow("    a, d : move forward/backward based on the robot's torso")
    print_yellow("    o, l:  rotate pitch lift/lower the gripper")

def get_new_episode_idx(task_demo_path):
    def extract_episode_idx(filename):
        numbers = re.findall(r'\d+', filename)  # Find all numbers
        return int(numbers[-1]) if numbers else 0  # Return the last one, or 0 if no number

    all_files = os.listdir(task_demo_path)
    if len(all_files) > 0:
        sorted_files = sorted(all_files, key=extract_episode_idx)
        last_ep_idx = extract_episode_idx(sorted_files[-1])
        new_ep_idx = int(last_ep_idx) + 1
    else:
        new_ep_idx = 1

    return new_ep_idx

def main():
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Exiting Teleoperation program.")
        sys.exit(0)  # Exit cleanly

    # Register SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    """ Definition for user's hyperparameters and constants """
    _dt = 0.4       # max = 1
    _dr = 0.5       # max = 1
    KEYBOARD_ACTION_MAP = {
        "d": np.array([-_dt, 0, 0, 0, 0, 0, -1], dtype=np.float32),
        "a": np.array([_dt, 0, 0, 0, 0, 0, -1], dtype=np.float32),
        "s": np.array([0, 0, -_dt, 0, 0, 0, -1], dtype=np.float32),
        "w": np.array([0, 0, _dt, 0, 0, 0, -1], dtype=np.float32),
        "o": np.array([0, 0, 0, 0, _dr, 0, -1], dtype=np.float32),
        "l": np.array([0, 0, 0, 0, -_dr, 0, -1], dtype=np.float32),
    }
    GRIPPER_STATE = {0: 'CLOSE', 1: 'OPEN'}

    """ Select tasks """
    task_name = 'jetrover-pickup-cube'
    env = JetRoverManipulationEnv()

    """ Utilities """
    def _execute_action(env, action):
        obs, reward, terminated, truncated, info = env.step(action)
        logger(obs, action, 0.0, 0.0, 0, None)
        print(f"Global step: {env.global_step}")

    def _execute_reset(env):
        obs = env.reset()
        action = np.array([0., 0., 0., 0., 0., 0., 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        logger(obs, action, 0.0, 0, None)
        print(f"Global step: {env.global_step}")

    def _get_gripper_state(env):
        raw_obs = env._get_observation()
        gripper_pos = raw_obs['state'][-1]
        gripper_state = 1.0 if gripper_pos >= 0.5 else 0.0
        return gripper_state

    """ Logger to store rollout data """
    root_demo_path = '/home/ubuntu/manipulation_experiment/teleoperation_dataset'
    task_demo_path = os.path.join(root_demo_path, task_name)
    if not os.path.exists(task_demo_path):
        os.makedirs(task_demo_path)

    filename_template = "{task_name}_episode_{ep_idx}.pkl"
    new_ep_idx = get_new_episode_idx(task_demo_path)
    filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
    logger = PickleLogger(filename=filename)

    """ Start Teleoperation """
    print_help()
    print("Started Teleoperation.")
    print(f"Current log's file: {logger.filename}")

    _execute_reset(env)

    running = True
    is_open = _get_gripper_state(env)     # The gripper is open at initial time
    while running:
        # Check for key press
        key = input("Next action (h to help): ")

        if key == "h":
            print_help()

        elif key == "q":
            print("Quitting teleoperation.")
            running = False
            continue

        elif key == "r":
            print("Resetting robot...")
            _execute_reset(env)
            new_ep_idx = get_new_episode_idx(task_demo_path)
            new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
            logger.make_new_rollout(filename=new_filename)
            is_open = 1
            print(f"Gripper is now: {GRIPPER_STATE[is_open]}")
            print_help()
            print(f"Current log's file: {logger.filename}")

        elif key == "m":
            logger.save()
            new_ep_idx = get_new_episode_idx(task_demo_path)
            new_filename = os.path.join(task_demo_path, filename_template.format(task_name=task_name, ep_idx=new_ep_idx))
            logger.make_new_rollout(filename=new_filename)
            print(f"New log's file: {logger.filename}")

        elif key == " ":
            is_open = 1 - is_open
            _execute_action(env, np.array([0., 0., 0., 0., 0., 0., is_open], dtype=np.float32))
            print(f"Gripper is now: {GRIPPER_STATE[is_open]}")

        else:
            action = np.zeros(7, dtype=np.float32)
            for char in key:
                if char not in KEYBOARD_ACTION_MAP:
                    print("Invalid key. Please try again.")
                    break
                action += KEYBOARD_ACTION_MAP[char]
            action[-1] = is_open
            _execute_action(env, action)

    print("Teleoperation ended.")

if __name__ == "__main__":
    main()
