import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

warnings.filterwarnings("ignore")
register(id="2048-v0", entry_point="envs:My2048Env")

my_config = {
    "run_id": "example",
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",
    "epoch_num": 5,
    "timesteps_per_epoch": 1000,
    "eval_episode_num": 10,
    "learning_rate": 1e-4,
}


def make_env():
    env = gym.make("2048-v0")
    return env


def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        env.seed(seed) # set seed using old Gym API
        obs = env.reset()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        avg_highest += info[0]["highest"]
        avg_score += info[0]["score"]

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num

    return avg_score, avg_highest


def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        is_better = current_best < avg_score

        print(f"epoch: {epoch} | avg score: {avg_score} | avg highest: {avg_highest} | best: {is_better}")
        wandb.log({"avg_highest": avg_highest, "avg_score": avg_score})

        if is_better:
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")


if __name__ == "__main__":
    run = wandb.init(
        project="rl-2048",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"],
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
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
    )

    train(eval_env, model, my_config)
