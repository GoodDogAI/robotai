import os
import gym
import time
import torch
import json
import glob
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from tqdm import tqdm

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure, HParam, Figure
from stable_baselines3 import PPO, DQN

from src.models.stable_baselines3.sac import CustomSAC
from src.models.stable_baselines3.env import MsgVecEnv
from src.models.stable_baselines3.feature_extractor import MsgVecNormalizeFeatureExtractor
from src.models.stable_baselines3.buffers import HostReplayBuffer

import src.train.reward_modifiers as reward_modifiers
from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.train.rldataset import MsgVecDataset
from stable_baselines3.common.buffers import ReplayBuffer


# TODO:
# - [X] Figure out refreshing caches if new data comes in while training
# - [X] Figure out why last few samples of that recent validation run are all the same value
# - [X] Check timings of loading messages, maybe its' device IO bottlenecked
# - [X] Normalize observations
# - [ ] Fill the buffer in a separate process
# - [X] Normalize rewards
# - [X] Delta on actions
# - [X] Record estimated target entropy in training
# - [X] What happens if msgvec actions are greater than 1.0, does the gradient explode? No, because we look at the gradient of tanh, not its inverse
# - [X] Do test runs with reward modifiers, and adjusting the reward to be less vision oriented
# - [ ] Balance out the manual reward and punishments so that they have roughly equal weight

def process_act_vector(act: np.ndarray, msgvec):
    if msgvec.is_discrete_act():
        # Sample using the distribution of the act vector
        return np.random.choice(np.arange(len(act)), p=act)
    else:
        return act

if __name__ == "__main__":
    brain_config = MODEL_CONFIGS["basic-brain-discrete-1"]
    log_dir = "/home/jake/robotai/_sb3_logs/"
    buffer_size = 50_000
    batch_size = 512
    net_arch = [256, 256]
    reward_modifier_fn = "reward_modifier_penalize_fast_move_backwards"
    validation_runname = "alphalog-4dc23143"  
    validation_buffer_size = 10_000
    num_updates = round(buffer_size * 10 / batch_size)

    msgvec = PyMsgVec(brain_config["msgvec"], PyMessageTimingMode.REPLAY)
    cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR), brain_config, getattr(reward_modifiers, reward_modifier_fn))
    env = MsgVecEnv(msgvec)
    obs_means = torch.zeros(env.observation_space.shape, dtype=torch.float32, requires_grad=False).to("cuda")
    obs_stds = torch.zeros(env.observation_space.shape, dtype=torch.float32, requires_grad=False).to("cuda")
    reward_mean = 0.0
    reward_std = 0.0

    model = DQN("MlpPolicy", env, buffer_size=buffer_size, verbose=1, 
                #learning_rate=1e-4,
                policy_kwargs={
                    "net_arch": net_arch,
                    "features_extractor_class": MsgVecNormalizeFeatureExtractor,
                    "features_extractor_kwargs": {
                        "obs_means": obs_means,
                        "obs_stds": obs_stds,
                    },
                },
                replay_buffer_class=HostReplayBuffer,
                replay_buffer_kwargs={"handle_timeout_termination": False})
 
    run_name = None

    # If run_name is not set, just create the next highest run1, run2, etc.. in the folder
    if run_name is None:
        try:
            rundirs = glob.glob(os.path.join(log_dir, "run*"))
            max_run = max([int(os.path.basename(d).replace("run", "")) for d in rundirs])
            run_name = f"run{max_run + 1}"
        except ValueError:
            run_name = "run1"

    # Setup the logger
    logger = configure(os.path.join(log_dir, run_name), ["tensorboard"])
    model.set_logger(logger)

    # Log the hyperparameters
    # https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
    hparam_dict={
        "algorithm": model.__class__.__name__,
        "buffer_size": buffer_size,
        "net_arch": str(net_arch),
        "batch_size": batch_size,
        "num_updates": num_updates,
        "learning_rate": model.learning_rate,
        "validation_runname": validation_runname,
        "validation_buffer_size": validation_buffer_size,
        "reward_modifier_fn": reward_modifier_fn,
    }

    if hasattr(model, "target_entropy") and model.target_entropy is not None:
        hparam_dict["target_entropy"] = float(model.target_entropy)

    if hasattr(model, "ent_coef") and model.ent_coef is not None:
        hparam_dict["ent_coef"] = "auto" if model.ent_coef == "auto" else float(model.ent_coef)


    # Copy the current file to the log directory, as a reference
    with open(__file__, "r") as f:
        with open(os.path.join(log_dir, run_name, "train_script.py"), "w") as f2:
            f2.write(f.read())

    # Copy over the brain config and all submodel configs for references
    with open(os.path.join(log_dir, run_name, "brain_config.json"), "w") as f2:
        json.dump(brain_config, f2, indent=4)

    for submodel_type, submodel in brain_config["models"].items():
        with open(os.path.join(log_dir, run_name, f"{submodel}_config.json"), "w") as f2:
            json.dump(MODEL_CONFIGS[submodel], f2, indent=4)

    # Read through the whole dataset, calculating statistics, and filling the training replay buffer
    buffer = model.replay_buffer
    samples_added = 0
    num_episodes = 0
    num_positive_rewards = 0
    num_negative_rewards = 0
    num_nicely_matched_samples = 0
      
    for entry in tqdm(cache.generate_dataset(), desc="Replay buffer", total=cache.estimated_size()):
        if samples_added < buffer_size:
            buffer.add(obs=entry["obs"], action=process_act_vector(entry["act"], msgvec), reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)
        
        samples_added += 1

        if entry["done"]:
            num_episodes += 1

        if entry["reward_override"]:
            if entry["reward"] >= 0:
                num_positive_rewards += 1
            else:
                num_negative_rewards += 1

        if np.max(entry["act"]) > 0.99:
            num_nicely_matched_samples += 1

        if samples_added == 1:
            obs_means.copy_(torch.from_numpy(entry["obs"]).cuda())
            reward_mean = entry["reward"]
        else:
            delta = torch.from_numpy(entry["obs"]).cuda() - obs_means
            obs_means += delta / samples_added
            obs_stds += delta * (torch.from_numpy(entry["obs"]).cuda() - obs_means)

            delta = entry["reward"] - reward_mean
            reward_mean += delta / samples_added
            reward_std += delta * (entry["reward"] - reward_mean)

    obs_stds /= samples_added - 1
    obs_stds.sqrt_()

    reward_std /= samples_added - 1
    reward_std = math.sqrt(reward_std)
    buffer.normalize_reward(reward_mean, reward_std)

    print(f"Read {samples_added} dataset samples")
    print(f"Contains {num_episodes} episodes")
    print(f"Contains {num_positive_rewards} positive rewards")
    print(f"Contains {num_negative_rewards} negative rewards")
    print(f"Contains {num_nicely_matched_samples} nicely matched samples")

    hparam_dict["num_samples"] = samples_added
    hparam_dict["num_episodes"] = num_episodes
    hparam_dict["num_positive_rewards"] = num_positive_rewards
    hparam_dict["num_negative_rewards"] = num_negative_rewards

    logger.record("hparams", HParam(hparam_dict=hparam_dict, metric_dict={
        "train/actor_loss": 0,
        "train/critic_loss": 0,
        "validation/act_var": 0,
    }))

    # Fill the validation replay buffer
    validation_buffer = ReplayBuffer(buffer_size=validation_buffer_size,
                                     observation_space=env.observation_space, action_space=env.action_space, handle_timeout_termination=False)
    valgroup = next(g for g in cache.groups if g[0].get_runname() == validation_runname)

    for entry in cache.generate_log_group(valgroup, shuffle_within_group=False):
        validation_buffer.add(obs=entry["obs"], action=process_act_vector(entry["act"], msgvec), reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)

    print(f"Added {validation_buffer.size()} samples to the validation buffer from {validation_runname}")

    for step in range(1000*1000):
        step_start_time = time.perf_counter()
        model.train(gradient_steps=num_updates, batch_size=batch_size)
        gradient_end_time = time.perf_counter()

        print(f"[{run_name}] Trained {num_updates} steps in {gradient_end_time - step_start_time:.2f}s")
        logger.record("perf/gradient_time", gradient_end_time - step_start_time)

        # Run the actor against the entire validation buffer, and measure the variance of the actions
        # validation_acts = []
        # perturbed_acts = []
        # for i in range(0, validation_buffer.size() - batch_size, batch_size):
        #     obs = torch.from_numpy(validation_buffer.observations[i:i+batch_size, 0]).to(model.device)
        #     replay_act = torch.from_numpy(validation_buffer.actions[i:i+batch_size, 0]).to(model.device)
        #     actor_act = model.actor(obs, deterministic=True).detach()
        #     validation_acts.append(actor_act.cpu())

        #     # Perturb the observations and see how much the output changes
        #     perturbed_obs = obs + torch.randn_like(obs, device=obs.device) * obs_stds * 0.1
        #     perturbed_acts.append(model.actor(perturbed_obs, deterministic=True).detach().cpu())
        #     logger.record_mean("validation/perturbed_act_diff_mean", torch.mean(torch.abs(validation_acts[-1] - perturbed_acts[-1])).item())

        #     q_replay = model.critic(obs, replay_act)
        #     q_actor = model.critic(obs, actor_act)
        #     for q_r, q_a in zip(q_replay, q_actor):
        #         logger.record_mean("validation/q_diff_mean", torch.mean(q_a - q_r).detach().cpu().item())

        #         if i == 0:
        #             logger.record_mean(f"validation/q_mean", torch.mean(q_r).item())

        # # Calculate and report statistics against the validation buffer
        # validation_acts = torch.concat(validation_acts)
        # validation_act_mean = torch.mean(validation_acts, axis=0)
        # validation_act_var = torch.var(validation_acts, axis=0)
        # for i, act in enumerate(brain_config["msgvec"]["act"]):
        #     logger.record(f"validation/{act['path'].replace('.', '_')}_mean", validation_act_mean[i].item())
        #     logger.record(f"validation/{act['path'].replace('.', '_')}_var", validation_act_var[i].item())
        #     logger.record(f"validation/{act['path'].replace('.', '_')}_stddev", torch.sqrt(validation_act_var[i]).item())
        #     logger.record(f"validation/{act['path'].replace('.', '_')}_hist", validation_acts[:, i])            

        #     figure = plt.figure()
        #     figure.add_subplot().plot(validation_acts[:, i].numpy())
        #     logger.record(f"trajectory/{act['path'].replace('.', '_')}", Figure(figure, close=True))
        #     plt.close()


        # logger.record(f"validation/act_var", torch.mean(validation_act_var).item())

        # Each step, replace 50% of the replay buffer with new samples
        for entry in itertools.islice(cache.sample_dataset(), buffer_size // 2):
            buffer.add(obs=entry["obs"], action=process_act_vector(entry["act"], msgvec), reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)
            samples_added += 1
        
        refill_end_time = time.perf_counter()
        logger.record("perf/buffer_time", refill_end_time - gradient_end_time)
        print(f"[{run_name}] Refilled {buffer_size // 2} entries in {refill_end_time - gradient_end_time:.2f}s")

        logger.dump(step=step)

        if step % 20 == 0:
            model.save(f"/home/jake/robotai/_checkpoints/basic-brain-discrete-1-sb3-{run_name}.zip")
            print("Model saved")
