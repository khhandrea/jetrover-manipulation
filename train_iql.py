import sys

from tqdm import trange

sys.path.append("./iql")
from iql.log import Logger
from iql.IQL import IQL
from iql.utils import ReplayBuffer

# Base directories
WORK_DIR = "iql/results/"
DATA_DIR = "teleoperation_dataset/jetrover-pickup-cube/"

def main():
    # Hyperparameters
    expectile = 0.7
    temperature = 3.0
    tau = 0.005
    discount = 0.99
    max_timesteps = 1e6
    save_freq = 1e4
    batch_size = 64

    replay_buffer = ReplayBuffer(DATA_DIR)
    state_dim = replay_buffer.state_dim
    action_dim = replay_buffer.action_dim

    policy = IQL(state_dim=state_dim,
                 action_dim=action_dim,
                 expectile=expectile,
                 discount=discount,
                 tau=tau,
                 temperature=temperature)


    logger = Logger(WORK_DIR)

    for t in trange(int(max_timesteps)):
        policy.train(replay_buffer, batch_size, logger=logger)

if __name__ == "__main__":
    main()
