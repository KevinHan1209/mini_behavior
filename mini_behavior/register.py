"""
Environment registration helper.
"""
import gymnasium as gym

env_list = []


def register(id, entry_point, reward_threshold=0.95, kwargs=None):
    assert id.startswith("MiniGrid-")
    if id in env_list:
        return

    gym.envs.registration.register(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
        kwargs=kwargs,
    )
    env_list.append(id)
