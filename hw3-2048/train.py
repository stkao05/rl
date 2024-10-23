import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.logger import configure
import numpy as np

warnings.filterwarnings("ignore")
register(id="2048-v0", entry_point="envs:My2048Env")

my_config = {
    "run_id": "example",
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",
    "epoch_num": 100,
    "eval_episode_num": 100,
    "timesteps_per_epoch": 1000,
    "learning_rate": 1e-4,
}

def make_env():
    env = gym.make("2048-v0")
    return env


def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    score = []
    highest = []
    step_count = []
    illegal_count = []

    for seed in range(eval_episode_num):
        done = False
        env.seed(seed) # set seed using old Gym API
        obs = env.reset()
        count = 0

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            count += 1

        score.append(info[0]["score"])
        highest.append(info[0]["highest"])
        step_count.append(count)
        illegal_count.append(1 if info[0]["illegal_move"] else 0)

    stats = {
        "score_mean": np.mean(score),
        "score_median": np.median(score),
        "score_max": np.max(score),
        "score_std": np.std(score),
        "highest_mean": np.mean(highest),
        "highest_median": np.median(highest),
        "highest_max": np.max(highest),
        "highest_std": np.std(highest),
        "step_count_mean": np.mean(step_count),
        "step_count_median": np.median(step_count),
        "step_count_max": np.max(step_count),
        "step_count_std": np.std(step_count),
        "illegal_count": np.sum(illegal_count) / eval_episode_num
    }

    return stats


def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100, # unsure how to intepret it
                verbose=2,
            ),
        )

        stats = eval(eval_env, model, config["eval_episode_num"])
        is_better = current_best < stats["score_mean"]

        print_stats(epoch, stats)
        wandb.log(stats)

        # print(f"epoch: {epoch} | avg score: {avg_score} | avg highest: {avg_highest} | best: {is_better}")
        # wandb.log({"avg_highest": avg_highest, "avg_score": avg_score})

        if is_better:
            current_best = stats["score_mean"]
            save_path = config["save_path"]
            model.save(f"{save_path}")


def print_stats(epoch, stats):
    output = f"epoch: {epoch:<3} | "
    for key, value in stats.items():
        output += f"{key}: {value:4.1f} | "
    print(output)


if __name__ == "__main__":
    # logger = configure("/tmp/sb3_log/", ["stdout", "tensorboard"])

    run = wandb.init(
        project="rl-2048",
        config=my_config,
        # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # id=my_config["run_id"],
    )

    num_train_envs = 2
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])
    eval_env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        # tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
    )
    # model.set_logger(logger)

    train(eval_env, model, my_config)
