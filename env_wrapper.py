import gym
import numpy as np
from mini_behavior.utils.states_base import RelativeObjectState

class CustomObservationWrapper(gym.ObservationWrapper):
    """Converts environment observations into a flat vector format"""

    DEFAULT_RELATIVE_STATES = [
        'atsamelocation',
        'infovofrobot',
        'inleftreachofrobot',
        'inrightreachofrobot',
        'inside',
        'nextto',
        'inlefthandofrobot',
        'inrighthandofrobot',
    ]

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.get_obs_space()

    def observation(self, obs):
        # Convert observations to flat vector format
        if isinstance(obs, dict):
            return self.gen_obs()
            
        # Handle both single and batched observations
        if len(obs.shape) == 1:
            return self.gen_obs()
        else:
            # For vectorized environments, generate observations for each env
            return np.stack([self.gen_obs() for _ in range(obs.shape[0])])

    def _resolve_obj_position(self, env, obj):
        """
        Resolve an object's absolute grid position.
        Objects stored inside containers inherit the container's position.
        """
        if obj.cur_pos is not None:
            return obj.cur_pos

        name = obj.get_name()
        if 'gear' in name and 'gear_toy' in env.objs:
            return env.objs['gear_toy'][0].cur_pos
        if 'winnie' in name and 'winnie_cabinet' in env.objs:
            return env.objs['winnie_cabinet'][0].cur_pos
        if 'coin' in name and 'piggie_bank' in env.objs:
            return env.objs['piggie_bank'][0].cur_pos
        if 'cube' in name and 'cube_cabinet' in env.objs:
            return env.objs['cube_cabinet'][0].cur_pos
        return None

    def _collect_object_features(self, env, agent_pos):
        # Modified gen_obs for single environment
        obj_states = []
        for obj_type in env.objs.values():
            for obj in obj_type:
                cur_pos = self._resolve_obj_position(env, obj)
                if cur_pos is None:
                    rel_pos = (0.0, 0.0)
                else:
                    rel_pos = (
                        float(cur_pos[0] - agent_pos[0]),
                        float(cur_pos[1] - agent_pos[1]),
                    )
                obj_states.extend(rel_pos)
                for state_name, state in obj.states.items():
                    if isinstance(state, RelativeObjectState):
                        continue
                    if state_name in self.DEFAULT_RELATIVE_STATES:
                        continue
                    obj_states.append(1.0 if state.get_value(env) else 0.0)
        return obj_states

    def _gen_single_obs(self, env_idx=0):
        env = self.env.envs[env_idx] if hasattr(self.env, 'envs') else self.env
        agent_pos = np.array(env.agent_pos, dtype=np.float32)
        obj_states = self._collect_object_features(env, agent_pos)
        obs = [float(agent_pos[0]), float(agent_pos[1]), float(env.agent_dir)] + obj_states
        return np.array(obs, dtype=np.float32)

    def get_obs_space(self):
        width = getattr(self.env, "width", None)
        height = getattr(self.env, "height", None)
        fallback_dim = getattr(self.env, "room_width", None)
        dims = [d for d in [width, height, fallback_dim] if d is not None]
        max_dim = max(dims) if dims else 1

        lows = [0.0, 0.0, 0.0]
        highs = [
            float(width if width is not None else max_dim),
            float(height if height is not None else max_dim),
            3.0,
        ]

        agent_pos = np.zeros(2, dtype=np.float32)
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                # Two slots for relative x/y
                lows.extend([-float(max_dim), -float(max_dim)])
                highs.extend([float(max_dim), float(max_dim)])
                for state_name, state in obj.states.items():
                    if isinstance(state, RelativeObjectState):
                        continue
                    if state_name in self.DEFAULT_RELATIVE_STATES:
                        continue
                    lows.append(0.0)
                    highs.append(1.0)

        low_arr = np.array(lows, dtype=np.float32)
        high_arr = np.array(highs, dtype=np.float32)
        return gym.spaces.Box(low=low_arr, high=high_arr, dtype=np.float32)

    def gen_obs(self):
        env = self.env
        agent_pos = np.array(env.agent_pos, dtype=np.float32)
        obj_states = self._collect_object_features(env, agent_pos)
        obs = [float(agent_pos[0]), float(agent_pos[1]), float(env.agent_dir)] + obj_states
        return np.array(obs, dtype=np.float32)
