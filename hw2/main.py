import random
import numpy as np
import json
import wandb
import time

import wandb.plot

from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int, epsilon=EPSILON):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= epsilon,
            )
    reward, loss = policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

    return reward, loss

def run_SARSA(grid_world: GridWorld, iter_num: int, epsilon=EPSILON):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon=epsilon,
            )

    rewards, losses = policy_iteration.run(max_episode=iter_num)

    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

    return rewards, losses


def run_Q_Learning(grid_world: GridWorld, iter_num: int, epsilon=EPSILON):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon=epsilon,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    rewards, losses = policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

    return rewards, losses


def bias_variance():
    v_true = np.load("/Users/stevenkao/workspace/rl/hw2/sample_solutions/prediction_GT.npy")

    def bias_variance_estimate(algo):
        run_num = 50
        v_preds = np.zeros((run_num, grid_world.get_state_space())) # (run_num, state_size)
        for i in range(0, run_num):
            seed = i + 1
            v_preds[i] = algo(grid_world, seed)

        v_pred_mean = v_preds.mean(axis=0) # (state_size)
        bias = v_pred_mean - v_true # (state_size)
        var = ((v_preds - v_pred_mean)**2).mean(axis=0) # (state_size)

        return bias, var

    td_bias, td_var = bias_variance_estimate(run_TD_prediction)
    mc_bias, mc_var = bias_variance_estimate(run_MC_prediction)
    plt.close()

    print("td bias", td_bias)
    print("mc bias", mc_bias)
    print("td var", td_var)
    print("mc var", mc_var)

    run = wandb.init(project="rl-hw2", name="TD")
    wandb.log({'bias': wandb.plot.histogram(
        wandb.Table(data=[[_] for _ in td_bias], columns=["bias"]), "bias", title="Bias")})
    wandb.log({'var': wandb.plot.histogram(
        wandb.Table(data=[[_] for _ in td_var], columns=["var"]), "var", title="Variance")})
    run.finish()

    run = wandb.init(project="rl-hw2", name="MC")
    wandb.log({'bias': wandb.plot.histogram(
        wandb.Table(data=[[_] for _ in mc_bias], columns=["bias"]), "bias", title="Bias")})
    wandb.log({'var': wandb.plot.histogram(
        wandb.Table(data=[[_] for _ in mc_var], columns=["var"]), "var", title="Variance")})
    run.finish()

    plt.hist(td_bias, bins=10, color="red", alpha=0.7, label="TD bias", range=(-0.2, 0.4))
    plt.hist(mc_bias, bins=10, color="blue", alpha=0.7, label="MC bias", range=(-0.2, 0.4))
    plt.legend(loc='upper right')
    plt.xlabel('Bias')
    plt.ylabel('Frequency')
    plt.savefig("bias_hist.png")
    plt.close()

    plt.hist(td_var, bins=10, color="red", alpha=0.7, label="TD var", range=(0, 0.005))
    plt.hist(mc_var, bins=10, color="blue", alpha=0.7, label="MC var", range=(0, 0.005))
    plt.legend(loc='upper right')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.savefig("var_hist.png")
    plt.close()

if __name__ == "__main__":
    seed = 1
    grid_world = init_grid_world("maze.txt",INIT_POS)
    # 2-1
    run_MC_prediction(grid_world,seed)
    run_TD_prediction(grid_world,seed)
    run_NstepTD_prediction(grid_world,seed)

    # 2-2
    grid_world = init_grid_world("maze.txt")
    run_MC_policy_iteration(grid_world, 512000)
    run_SARSA(grid_world, 512000)
    run_Q_Learning(grid_world, 50000)


    ## -------------------------- ##

    plt.close()
    epsilons = [0.1, 0.2, 0.3, 0.4]
    names = ["mc", "sarsa", "q"]
    grid_world = init_grid_world("maze.txt")

    def plot_learning(name, run_func, iteration, epsilons=None):
        if not epsilons:
            epsilons = [0.1, 0.2, 0.3, 0.4]

        for e in epsilons:
            start_time = time.time()
            losses, rewards = run_func(grid_world, iteration, epsilon=e)
            end_time = time.time()

            np.save(f"output/{name}-{e}-reward", rewards)
            np.save(f"output/{name}-{e}-loss", losses)
            print(f"{name}-{e}: {end_time - start_time:.2f} seconds")

    # plot_learning("mc", run_MC_policy_iteration, 512000, epsilons=[0.1])
    # plot_learning("mc", run_MC_policy_iteration, 512000, epsilons=[0.2, 0.3, 0.4])
    # plot_learning("sarsa", run_SARSA, 512000)
    # plot_learning("q", run_Q_Learning, 512000)

    # --------- #

    # import os
    # os.mkdir("figs")


    # --- per espilon ------ #
    # color = {
    #     "mc": "deepskyblue",
    #     "sarsa": "orange",
    #     "q": "green",
    # }

    # plt.close()
    # for type in ["loss", "reward"]:
    #     for e in epsilons:
    #         for name in names:
    #             # if name == "mc" and e == 0.1:
    #             #     continue
    #             items = np.load(f"output/{name}-{e}-{type}.npy")

    #             n = 20
    #             if len(items) % n != 0:
    #                 items = items[0: n * (len(items) // n)]

    #             items = items.reshape(-1, n).mean(axis=1)[::100]
    #             plt.plot(items, label=f"{name}", alpha=0.7, color=color[name])

    #         plt.title(f"{type} (epsilon = {e})")
    #         plt.legend(loc="upper right")
    #         plt.savefig(f"figs/episolon-{e}-{type}.png")
    #         plt.close()


    # --- per algo ------ #
    # plt.close()

    # for type in ["loss", "reward"]:
    #     for name in ["q"]:
    #         for e in epsilons:
    #             if name == "mc" and e == 0.1:
    #                 continue
    #             items = np.load(f"output/{name}-{e}-{type}.npy")
    #             n = 10
    #             if len(items) % n != 0:
    #                 items = items[0: n * (len(items) // n)]

    #             items = items.reshape(-1, n).mean(axis=1)[::100]
    #             plt.plot(items, label=f"e={e}", alpha=0.7)

    #         plt.title(f"{type} for {name}")
    #         plt.legend()
    #         plt.savefig(f"figs/{name}-{type}.png")
    #         plt.close()

    # name = "q"
    # for e in epsilons:
    #     losses = np.load(f"output/{name}-{e}-loss.npy")
    #     reward = np.load(f"output/{name}-{e}-reward.npy")
    #     print(losses.shape, reward.shape)

    # type = "loss"
    # name = "q"
    # for e in [0.2, 0.3, 0.4]:
    #     if name == "mc" and e == 0.1:
    #         continue
    #     items = np.load(f"output/{name}-{e}-{type}.npy")
    #     n = 10
    #     if len(items) % n != 0:
    #         items = items[0: n * (len(items) // n)]

    #     items = items.reshape(-1, n).mean(axis=1)[200:-1:100]
    #     plt.plot(items, label=f"e={e}", alpha=0.7)

    # plt.title(f"{type} for {name}")
    # plt.legend(loc="upper right")
    # plt.savefig(f"figs/{name}-{type}.png")
    # plt.close()
    # import code; code.interact(local=locals())
