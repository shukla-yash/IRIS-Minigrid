from typing import Any, Tuple

import numpy as np
from PIL import Image

from .done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        obs, _ = self.env.reset()
        obs = self.resize(obs['image'])
        return obs[None, ...]

    def step(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, reward, done, done2, _ = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        done = np.array([done])
        self.update_done_tracker(done)
        obs = self.resize(obs['image'])
        return obs[None, ...], np.array([reward]), done, None

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize((64,64), Image.BILINEAR)
        return np.array(img)
