import os
import gym
import time
import torch
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, HParam, Figure
from torch.profiler import profile, record_function, ProfilerActivity

from src.config import HOST_CONFIG, MODEL_CONFIGS
from src.msgvec.pymsgvec import PyMsgVec, PyTimeoutResult, PyMessageTimingMode
from src.train.rldataset import MsgVecDataset
from stable_baselines3.common.buffers import ReplayBuffer
from src.train.stable_baselines_buffers import HostReplayBuffer


class MsgVecEnv(gym.Env):
    def __init__(self, msgvec) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(msgvec.act_size(),))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(msgvec.obs_size(),))


# TODO:
# - [ ] Figure out refreshing caches if new data comes in while training
# - [ ] Figure out why last few samples of that recent validation run are all the same value
# - [ ] Check timings of loading messages, maybe its' device IO bottlenecked
# - [ ] Normalize observations
# - [ ] Normalize rewards
# - [ ] Delta on rewards


if __name__ == "__main__":
    brain_config = MODEL_CONFIGS["basic-brain-test1"]
    msgvec = PyMsgVec(brain_config["msgvec"], PyMessageTimingMode.REPLAY)
    cache = MsgVecDataset(os.path.join(HOST_CONFIG.RECORD_DIR), brain_config)
    log_dir = "/home/jake/robotai/_sb3_logs/"
    buffer_size = 50_000
    batch_size = 512
    validation_runname = "alphalog-4425c446"  
    validation_buffer_size = 10_000
    num_updates = round(buffer_size * 10 / batch_size)

    env = MsgVecEnv(msgvec)
    model = SAC("MlpPolicy", env, buffer_size=buffer_size, verbose=1, 
                ent_coef=0.95,
                learning_rate=1e-4,
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
        "batch_size": batch_size,
        "num_updates": num_updates,
        "learning_rate": model.learning_rate,
        "validation_runname": validation_runname,
        "validation_buffer_size": validation_buffer_size,
    }

    if model.target_entropy is not None:
        hparam_dict["target_entropy"] = model.target_entropy

    if model.ent_coef is not None:
        hparam_dict["ent_coef"] = model.ent_coef

    logger.record("hparams", HParam(hparam_dict=hparam_dict, metric_dict={
        "train/actor_loss": 0,
        "train/critic_loss": 0,
        "validation/act_var": 0,
    }))

    # Copy the current file to the log directory, as a reference
    with open(__file__, "r") as f:
        with open(os.path.join(log_dir, run_name, "train_script.py"), "w") as f2:
            f2.write(f.read())

    # Fill the training replay buffer
    buffer = model.replay_buffer
    samples_added = 0
      
    for entry in tqdm(itertools.islice(cache.generate_samples(), buffer_size), desc="Replay buffer", total=buffer_size):
        buffer.add(obs=entry["obs"], action=entry["act"], reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)
        samples_added += 1

    print(f"Added {samples_added} samples to the replay buffer")

    # Fill the validation replay buffer
    validation_buffer = ReplayBuffer(buffer_size=validation_buffer_size,
                                     observation_space=env.observation_space, action_space=env.action_space, handle_timeout_termination=False)
    groups = cache.lh.group_logs()
    valgroup = next(g for g in groups if g[0].get_runname() == validation_runname)

    for entry in cache.generate_log_group(valgroup, shuffle_within_group=False):
        validation_buffer.add(obs=entry["obs"], action=entry["act"], reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)

    observation_stds = torch.from_numpy(validation_buffer.observations[:validation_buffer.size(), 0].std(axis=0)).to(model.device)

    print(f"Added {validation_buffer.size()} samples to the validation buffer from {validation_runname}")

    for step in range(1000*1000):
        step_start_time = time.perf_counter()
        model.train(gradient_steps=num_updates, batch_size=batch_size)
        gradient_end_time = time.perf_counter()

        print(f"[{run_name}] Trained {num_updates} steps in {gradient_end_time - step_start_time:.2f}s")
        logger.record("perf/gradient_time", gradient_end_time - step_start_time)

        # Run the actor against the entire validation buffer, and measure the variance of the actions
        validation_acts = []
        perturbed_acts = []
        for i in range(0, validation_buffer.size(), batch_size):
            obs = torch.from_numpy(validation_buffer.observations[i:i+batch_size, 0]).to(model.device)
            validation_acts.append(model.actor(obs, deterministic=True).detach().cpu())

            # Perturb the observations and see how much the output changes
            perturbed_obs = obs + torch.randn_like(obs, device=obs.device) * observation_stds * 0.1
            perturbed_acts.append(model.actor(perturbed_obs, deterministic=True).detach().cpu())
            logger.record_mean("validation/perturbed_act_diff_mean", torch.mean(torch.abs(validation_acts[-1] - perturbed_acts[-1])).item())

        # Calculate and report statistics against the validation buffer
        validation_acts = torch.concat(validation_acts)
        validation_act_mean = torch.mean(validation_acts, axis=0)
        validation_act_var = torch.var(validation_acts, axis=0)
        for i, act in enumerate(brain_config["msgvec"]["act"]):
            logger.record(f"validation/{act['path'].replace('.', '_')}_mean", validation_act_mean[i].item())
            logger.record(f"validation/{act['path'].replace('.', '_')}_var", validation_act_var[i].item())
            logger.record(f"validation/{act['path'].replace('.', '_')}_stddev", torch.sqrt(validation_act_var[i]).item())
            logger.record(f"validation/{act['path'].replace('.', '_')}_hist", validation_acts[:, i])            

            figure = plt.figure()
            figure.add_subplot().plot(validation_acts[:, i].numpy())
            logger.record(f"trajectory/{act['path'].replace('.', '_')}", Figure(figure, close=True))
            plt.close()


        logger.record(f"validation/act_var", torch.mean(validation_act_var).item())

        # Each step, replace 50% of the replay buffer with new samples
        for entry in itertools.islice(cache.generate_samples(), buffer_size // 2):
            buffer.add(obs=entry["obs"], action=entry["act"], reward=entry["reward"], next_obs=entry["next_obs"], done=entry["done"], infos=None)
            samples_added += 1
        
        refill_end_time = time.perf_counter()
        logger.record("perf/buffer_time", refill_end_time - gradient_end_time)
        print(f"[{run_name}] Refilled {buffer_size // 2} entries in {refill_end_time - gradient_end_time:.2f}s")

        logger.dump(step=step)

        if step % 20 == 0:
            model.save(f"/home/jake/robotai/_checkpoints/basic-brain-test1-sb3-{run_name}.zip")
            print("Model saved")