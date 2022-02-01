from torch import float32
from carl.circuit import Circuit
from carl.environment import Environment
import learnrl as rl
from numpy import argmax
import tensorflow as tf
import os

kl = tf.keras.layers


class TFmemory():
    def __init__(self, max_memory_len=1000):
        self.max_memory_len = max_memory_len
        self.memory_len = 0
        self.MEMORY_KEYS = ('observation', 'action', 'reward',
                            'done', 'next_observation')
        self.datas = {key: None for key in self.MEMORY_KEYS}

    def remember(self, observation, action, reward,
                 done, next_observation):

        for val, key in zip((observation, action, reward,
                             done, next_observation), self.MEMORY_KEYS):

            batched_val = tf.expand_dims(val, axis=0)

            if self.memory_len == 0:
                self.datas[key] = batched_val
            else:
                self.datas[key] = tf.concat(
                    (self.datas[key], batched_val), axis=0)
            self.datas[key] = self.datas[key][-self.max_memory_len:]

        self.memory_len = len(self.datas[self.MEMORY_KEYS[0]])

    def sample(self, sample_size, method='random'):
        if method == 'random':
            indices = tf.random.shuffle(tf.range(self.memory_len))[
                :sample_size]
            datas = [tf.gather(self.datas[key], indices)
                     for key in self.MEMORY_KEYS]
        elif method == 'last':
            datas = [self.datas[key][-sample_size:]
                     for key in self.MEMORY_KEYS]
        else:
            raise NotImplementedError(f'Unknown sampling method {method}')

        return datas

    def __len__(self):
        return self.memory_len


class Evaluation():
    def __init__(self, discount=0.99):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError(
            'You must use eval only when subclassing Evaluation')

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)


class QLearning(Evaluation):
    def eval(self, rewards, dones, next_observations, action_value):
        future_rewards = rewards

        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            Q_max = tf.reduce_max(action_value(
                next_observations[ndones]), axis=-1)
            ndones_indices = tf.where(ndones)
            future_rewards = tf.tensor_scatter_nd_add(
                future_rewards, ndones_indices, self.discount * Q_max)
        return future_rewards


class Control():
    def __init__(self, action_space, exploration, exploration_decay=0):
        self.action_space = action_space
        self.exploration = exploration
        self.exploration_decay = exploration_decay

    def update_exploration(self):
        self.exploration *= (1-self.exploration_decay)

    def act(self, Q):
        raise NotImplementedError(
            'You must redefine act(self, Q) when subclassing Control')

    def __call__(self, Q, greedy=False):
        if greedy:
            actions = tf.argmax(Q, axis=-1)
        else:
            actions = self.act(Q)
        return actions


class Greedy(Control):
    def act(self, Q):
        greedy_actions = tf.argmax(Q, axis=-1, output_type=tf.int32)

        batch_size = Q.shape[0]
        action_size = self.action_space.n
        random_actions = tf.random.uniform(
            (batch_size,), minval=0, maxval=action_size, dtype=tf.int32)

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd <= self.exploration,
                           random_actions, greedy_actions)

        return actions


class DQAgent(rl.Agent):
    def __init__(self, action_space, action_value: tf.keras.Model = None, control: Control = None, evaluation: Evaluation = None, memory: TFmemory = None, sample_size=32, learning_rate=1e-3, learning_rate_decay=1e-6):
        self.action_value = action_value

        self.action_value_opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, decay=learning_rate_decay)

        self.control = Greedy(action_space, 0) if control is None else control
        self.evaluation = evaluation
        self.memory = memory
        self.sample_size = sample_size

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)  # [1,*dim_obs]
        Q = self.action_value(observations)  # [1,n_actions]
        actions = self.control(Q, greedy)
        return actions[0]

    def learn(self):

        if len(self.memory) < self.sample_size:
            return {}

        observations, actions, rewards, dones, next_observations = self.memory.sample(
            self.sample_size)

        expected_futur_rewards = self.evaluation(
            rewards, dones, next_observations, self.action_value)

        with tf.GradientTape() as tape:
            Q = self.action_value(observations, training=True)

            action_indices = tf.stack(
                (tf.range(len(actions)), actions), axis=-1)

            Q_actions = tf.gather_nd(Q, action_indices)

            loss = tf.keras.losses.mean_squared_error(
                expected_futur_rewards, Q_actions)

        grads = tape.gradient(loss, self.action_value.trainable_weights)

        self.action_value_opt.apply_gradients(
            zip(grads, self.action_value.trainable_weights))

        metrics = {
            'loss': loss.numpy(),
            'exploration': self.control.exploration,
            'learning_rate': self.action_value_opt._decayed_lr(tf.float32),
        }
        self.control.update_exploration()
        return metrics

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **kwargs):
        self.memory.remember(observation, action, reward,
                             done, next_observation)

    def save(self, filename):
        tf.keras.models.save_model(
            self.action_value, filename, save_format='h5')

    def load(self, filename):
        self.action_value = tf.keras.models.load_model(filename)


class CheckpointCallback(rl.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_episodes_cycle_end(self, episode: int, logs: dict = None):
        self.playground.agents[0].save(self.filename)
        # return super().on_episodes_cycle_end(episode, logs)


class Validationcallback(rl.Callback):
    def on_episodes_cycle_end(self, episode: int, logs: dict = None):
        self.playground.test(1)
        # return super().on_episodes_cycle_end(episode, logs)


if __name__ == "__main__":
    circuit = Circuit([(0, 0), (0.5, 1), (0, 2), (2, 2),
                       (3, 1), (6, 2), (6, 0)], n_cars=1, width=0.3)
    env = Environment(circuit)

    control = Greedy(env.action_space, 0.1, 1e-3)
    evaluation = QLearning()
    memory = TFmemory()

    action_value = tf.keras.models.Sequential((
        kl.Dense(64, activation='relu'),
        kl.Dense(64, activation='relu'),
        kl.Dense(env.action_space.n, activation='linear'),
    ))

    agent = DQAgent(
        env.action_space,
        action_value=action_value,
        control=control,
        evaluation=evaluation,
        memory=memory
    )

    checkpoint = CheckpointCallback(os.path.join('models', 'nouveau_model.h5'))
    valid = Validationcallback()

    pg = rl.Playground(env, agent)
    pg.fit(5000, verbose=1, callbacks=[checkpoint, valid])
    pg.test(1)
