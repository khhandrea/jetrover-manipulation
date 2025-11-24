import sys

import torch

sys.path.append("./iql")
from iql.IQL import IQL
from jetrover_gym.manipulation_env import JetRoverManipulationEnv


MODEL_PATH = "iql/results/models/bc_actor_s1000000.pth"

def main():
    # Hyperparameters
    expectile = 0.7
    temperature = 3.0
    tau = 0.005
    discount = 0.99
    max_timesteps = 100

    env = JetRoverManipulationEnv()
    state_dim = env.observation_space
    action_dim = env.action_space

    obs, _ = env.reset()

    state_dim = 7
    action_dim = 7

    policy = IQL(state_dim=state_dim,
                 action_dim=action_dim,
                 expectile=expectile,
                 discount=discount,
                 tau=tau,
                 temperature=temperature)

    policy.actor.load_state_dict(torch.load(MODEL_PATH))

    for _ in range(max_timesteps):
        obs = obs['state']
        action = policy.select_action(obs)
        obs, _, _, _, _ = env.step(action)
        print("ACTION:", action)
    env.close()

if __name__ == "__main__":
    main()
