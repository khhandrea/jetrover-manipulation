import copy
import os
import random
import re

import numpy as np
import pickle
from PIL import Image
import torch
from tqdm import tqdm

from iql.resnet import Resnet

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))


class ReplayBuffer(object):
    def __init__(self, dataset_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._postfix_process = "processed"

        self.state_dim = None
        self.action_dim = None
        self._all_episodes = self._preprocess_data(dataset_dir=dataset_dir)
        self._prepare_training_data()

    def _preprocess_data(self, dataset_dir):
        """
        Convert each episode file into one pickle file.
        """
        dataset_dir = dataset_dir[:-1] if dataset_dir[-1] == "/" else dataset_dir
        parent_path = os.path.dirname(dataset_dir)
        directory_name = dataset_dir.split("/")[-1]
        filename = directory_name + f"_{self._postfix_process}.pkl"
        preprocess_filename = os.path.join(parent_path, filename)

        if os.path.exists(preprocess_filename):
            print(f"Loading pre-processed dataset: {preprocess_filename}")
            with open(preprocess_filename, 'rb') as f:
                processed_dataset = pickle.load(f)
            return processed_dataset
        else:
            all_files = os.listdir(dataset_dir)
            sorted_files = sorted(all_files, key=extract_episode_idx)
            n_demos = len(sorted_files)

            all_raw_data = []
            for i in range(n_demos):
                with open(os.path.join(dataset_dir, sorted_files[i]), "rb") as f:
                    data = pickle.load(f)
                all_raw_data.append(data)

            print(f"Pre-processing dataset: {preprocess_filename}")
            resnet_args = AttrDict(visual_encoder="resnet18_r3m", gpu=True)
            feature_extractor = Resnet(resnet_args, eval=True, use_conv_feat=False)

            raw_all_episodes = []
            for ep_idx in tqdm(range(n_demos)):
                ep_len = len(all_raw_data[ep_idx])
                cur_episode = []
                for time_idx in range(ep_len - 1):
                    step_t = all_raw_data[ep_idx][time_idx]
                    step_tp1 = all_raw_data[ep_idx][time_idx + 1]

                    # Convert images to features
                    rgb_image_t = step_t['observation']['rgb']
                    rgb_image_tp1 = step_tp1['observation']['rgb']
                    img_feats = feature_extractor.featurize(
                        [Image.fromarray(rgb_image_t), Image.fromarray(rgb_image_tp1)],
                        batch=1
                    ).cpu().squeeze().numpy()

                    state_t = step_t['observation']['state'][:7]
                    state_tp1 = step_tp1['observation']['state'][:7]
                    action_t = step_t['action']
                    reward_t = step_tp1['reward']
                    done_t = step_tp1['terminated'] or step_tp1['truncated']
                    step_processed = (state_t, action_t, state_tp1, reward_t, done_t)
                    cur_episode.append(step_processed)
                step_last = (state_tp1, action_t, state_tp1, reward_t, done_t)
                cur_episode.append(step_last)

                raw_all_episodes.append(cur_episode)

            # convert all episodes to dict
            all_episodes = []
            all_episode_lens = []
            for ep_idx in range(len(raw_all_episodes)):
                cur_ep = {'observation': [], 'action': [], 'next_observation': [], 'reward': [], 'done': []}
                cur_ep_data = raw_all_episodes[ep_idx]
                ep_len = len(cur_ep_data)
                all_episode_lens.append(ep_len)
                for i in range(ep_len):
                    cur_ep['observation'].append(cur_ep_data[i][0])
                    cur_ep['action'].append(cur_ep_data[i][1])
                    cur_ep['next_observation'].append(cur_ep_data[i][2])
                    cur_ep['reward'].append(cur_ep_data[i][3])
                    cur_ep['done'].append(cur_ep_data[i][4])

                cur_ep['observation'] = np.array(cur_ep['observation'])
                cur_ep['action'] = np.array(cur_ep['action'])
                cur_ep['next_observation'] = np.array(cur_ep['next_observation'])
                cur_ep['reward'] = np.array(cur_ep['reward'])
                cur_ep['done'] = np.array(cur_ep['done'])

                all_episodes.append(copy.deepcopy(cur_ep))

            with open(preprocess_filename, "wb") as f:
                pickle.dump(all_episodes, f)
            return all_episodes

    def _prepare_training_data(self, action_eps=1e-5):
        """
        Load preprocessed data and add reward to episodes
        """
        self.state_dim = self._all_episodes[0]['observation'][0].shape[0]
        self.action_dim = self._all_episodes[0]['action'][0].shape[0]

        self._n_episodes = len(self._all_episodes)
        all_episode_lens = []
        for ep_idx in range(len(self._all_episodes)):
            all_episode_lens.append(self._all_episodes[ep_idx]['observation'].shape[0])
        self._all_episode_lens = np.array(all_episode_lens)

        # Modify reward for sparse
        for ep_idx in range(len(self._all_episodes)):
            for t in range(self._all_episode_lens[ep_idx]):
                state_t = self._all_episodes[ep_idx]['observation'][t]
                action_t = self._all_episodes[ep_idx]['action'][t]
                state_tp1 = self._all_episodes[ep_idx]['next_observation'][t]

                # Spare reward based on the gripper's state
                if state_tp1[-1] == 1: # Open
                    reward_t = -1.0
                elif state_tp1[-1] == 0: # Close
                    reward_t = 0.0
                else:
                    raise ValueError

                # Always done at last state of episode
                if t < self._all_episode_lens[ep_idx] - 1:
                    done_t = 0.0
                else:
                    done_t = 1.0
                # self._all_episodes[ep_idx]['reward'][t] = reward_t
                self._all_episodes[ep_idx]['done'][t] = done_t

        print(f"Number of episodes: {self._n_episodes}")
        print(f"Min horizon: {np.min(self._all_episode_lens)}")
        print(f"Max horizon: {np.max(self._all_episode_lens)}")
        print(f"Avg. horizon: {np.mean(self._all_episode_lens)}")
        print(f"Total transitions: {np.sum(self._all_episode_lens)}")


    def sample(self, batch_size):
        ep_idxes = np.random.randint(0, len(self._all_episode_lens), size=batch_size)
        offsets = np.random.randint(0, self._all_episode_lens[ep_idxes])

        state = []
        next_state = []
        action = []
        reward = []
        not_done = []
        for i in range(batch_size):
            ep_idx, timestep = ep_idxes[i], offsets[i]
            state.append([self._all_episodes[ep_idx]['observation'][timestep]])
            action.append([self._all_episodes[ep_idx]['action'][timestep]])
            next_state.append([self._all_episodes[ep_idx]['next_observation'][timestep]])
            reward.append([self._all_episodes[ep_idx]['reward'][timestep]])
            not_done.append([1 - self._all_episodes[ep_idx]['done'][timestep]])

        state = np.concatenate(state, axis=0)
        next_state = np.concatenate(next_state, axis=0)
        action = np.concatenate(action, axis=0)
        reward = np.concatenate(reward, axis=0)
        not_done = np.concatenate(not_done, axis=0)

        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(not_done).to(self.device)
        )


def extract_episode_idx(filename):
    numbers = re.findall(r'\d+', filename)  # Find all numbers
    return int(numbers[-1]) if numbers else 0  # Return the last one, or 0 if no number


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "r3m.R3M"
    config["device"] = device
    
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    # config.agent["langweight"] = 0
    return config.agent


def remove_language_head(state_dict):
    keys = state_dict.keys()
    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    for key in list(keys):
        if ("lang_enc" in key) or ("lang_rew" in key):
            del state_dict[key]
    return state_dict