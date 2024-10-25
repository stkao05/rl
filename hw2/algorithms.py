import numpy as np
import json
from collections import deque, defaultdict

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method

        init_state = self.grid_world.reset()
        mc_returns = defaultdict(list)

        while self.episode_counter < self.max_episode:
            # new episode
            episode = []
            done = False
            cur_state = init_state
            first_appear = defaultdict(int) # at which timestep a state first appear

            while not done:
                if cur_state not in first_appear:
                    first_appear[cur_state] = len(episode)

                next_state, reward, done = self.collect_data()
                episode.append((cur_state, reward))
                cur_state = next_state

            # episode has end. the next_state would return the init state for new episode
            init_state = next_state 

            retrn = 0
            for t, (cur_state, reward) in reversed(list(enumerate(episode))):
                retrn = self.discount_factor * retrn + reward
                if first_appear[cur_state] < t:
                    continue
                mc_returns[cur_state].append(retrn)
                self.values[cur_state] = np.mean(mc_returns[cur_state])


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm

        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            next_state, reward, done = self.collect_data()
            cur_val = self.values[current_state]

            if done:
                td_target = reward
            else:
                td_target = reward + self.discount_factor * self.values[next_state]

            self.values[current_state] = cur_val + self.lr * (td_target - cur_val)
            current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        history = []
        n = self.n

        def backup(n_steps, done, next_state):
            steps = history[-n_steps:]
            rewards = [r for _, r in steps]
            s, _ = steps[0]
            next_state_value = 0 if done else self.values[next_state]
            td_target = (
                sum([r * self.discount_factor**i for i, r in enumerate(rewards)])
                + self.discount_factor ** len(steps) * next_state_value
            )
            self.values[s] = self.values[s] + self.lr * (td_target - self.values[s])

        while self.episode_counter < self.max_episode:
            next_state, reward, done = self.collect_data()
            history.append((current_state, reward))
            history = history[-n:]

            if len(history) == n:
                backup(n, done, next_state)

            if done:
                for i in range(1, min(n, len(history))):
                    backup(i, done, next_state)
                history = []

            current_state = next_state


# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        
        raise NotImplementedError
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy

        raise NotImplementedError


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        ep_rewards = []
        ep_loss = []

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # if iter_episode % 1000 == 0:
            #     print(f"{iter_episode / max_episode * 100:.2f}")

            history = []
            loss_trace = []
            done = False

            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_space)
                else:
                    action = self.q_values[current_state].argmax()

                next_state, reward , done = self.grid_world.step(action)
                history.append((current_state, action, reward))
                current_state = next_state

            n = len(history)
            # if n > 100000:
            #     print("large history", n)

            g = 0
            for t in reversed(range(n)):
                state, action, reward  = history[t]
                g = self.discount_factor * g + reward
                error = g - self.q_values[state][action]
                self.q_values[state][action] += self.lr * error
                # loss_trace.append(abs(error))

            # ep_loss.append(np.mean(loss_trace))
            # ep_rewards.append(np.mean(reward_trace))
            iter_episode += 1

        return ep_rewards, ep_loss


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy

        def choose_action(state):
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = self.q_values[state].argmax()
            return action

        iter_episode = 0
        current_state = self.grid_world.reset()
        action = choose_action(current_state)
        ep_reward = []
        ep_loss = []

        while iter_episode < max_episode:
            # if iter_episode % 30000 == 0:
            #     print(f"{iter_episode / max_episode * 100:.2f}")

            done = False
            reward_trace = []
            loss_trace = []

            while not done:
                next_state, reward, done = self.grid_world.step(action)
                next_action = choose_action(next_state)
                td_target = reward if done else reward + self.discount_factor * self.q_values[next_state][next_action]
                td_error = td_target - self.q_values[current_state][action]
                self.q_values[current_state][action] += self.lr * td_error

                # reward_trace.append(reward)
                # loss_trace.append(abs(td_error))
                action = next_action
                current_state = next_state

            # ep_reward.append(np.mean(reward_trace))
            # ep_loss.append(np.mean(loss_trace))
            iter_episode += 1

        return ep_reward, ep_loss



class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        raise NotImplementedError

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        ep_rewards = []
        ep_loss = []

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # if iter_episode % 1000 == 0:
            #     print(f"{iter_episode / max_episode * 100:.2f}: {len(ep_rewards)} / {iter_episode} / {max_episode}")

            done = False
            reward_trace = []
            loss_trace = []

            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_space)
                else:
                    action = self.q_values[current_state].argmax()

                next_state, reward, done = self.grid_world.step(action)
                self.buffer.append((current_state, action, next_state, reward, done))
                # reward_trace.append(reward)
                current_state = next_state

                if len(self.buffer) % self.update_frequency == 0:
                    for _ in range(self.sample_batch_size):
                        (s, a, ss, r, d) = self.buffer[np.random.randint(0, len(self.buffer))]
                        if d:
                            td_error = r - self.q_values[s][a]
                        else:
                            td_error = r + self.discount_factor * self.q_values[ss].max() - self.q_values[s][a]

                        self.q_values[s][a] += self.lr * (td_error)
                        # loss_trace.append(abs(td_error))

            # ep_rewards.append(np.mean(reward_trace))
            # if len(loss_trace) > 0: # first few episode might not have enough item in the buffer to trigger the update
            #     ep_loss.append(np.mean(loss_trace))

            iter_episode += 1

        return ep_rewards, ep_loss
