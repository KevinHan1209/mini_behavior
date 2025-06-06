import gym
import numpy as np
from mini_behavior.utils.states_base import RelativeObjectState

class CustomObservationWrapper(gym.ObservationWrapper):
    """Converts environment observations into a flat vector format"""
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

    def _gen_single_obs(self, env_idx=0):
        default_states = [
            'atsamelocation',
            'infovofrobot',
            'inleftreachofrobot',
            'inrightreachofrobot',
            'inside',
            'nextto',
        ]
        # Modified gen_obs for single environment
        obj_states = []
        env = self.env.envs[env_idx] if hasattr(self.env, 'envs') else self.env
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                # Some objects are attached to others or inside others, and are off the grid in code but should be in same position as parent object
                if obj.cur_pos is None:
                    if 'gear' in obj.get_name():
                        cur_pos = self.env.objs['gear_toy'][0].cur_pos
                    elif 'winnie' in obj.get_name():
                        cur_pos = self.env.objs['winnie_cabinet'][0].cur_pos
                    elif 'coin' in obj.get_name():
                        cur_pos = self.env.objs['piggie_bank'][0].cur_pos
                    elif 'cube' in obj.get_name():
                        cur_pos = self.env.objs['cube_cabinet'][0].cur_pos
                else:
                    cur_pos = obj.cur_pos
                obj_states += [*cur_pos]
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        if state_value not in default_states:
                            state = obj.states[state_value].get_value(self.env)
                            obj_states.append(1 if state else 0)
                        else:
                            continue

        obs = list(env.agent_pos) + [env.agent_dir] + obj_states
        return np.array(obs, dtype=np.float32)

    def get_obs_space(self):
        default_states = [
            'atsamelocation',
            'infovofrobot',
            'inleftreachofrobot',
            'inrightreachofrobot',
            'inside',
            'nextto',
        ]
        # Count non-relative object states and add agent position (x,y,dir)
        obj_states = []
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                # Some objects are attached to others or inside others, and are off the grid in code but should be in same position as parent object
                if obj.cur_pos is None:
                    if 'gear' in obj.get_name():
                        cur_pos = self.env.objs['gear_toy'][0].cur_pos
                    elif 'winnie' in obj.get_name():
                        cur_pos = self.env.objs['winnie_cabinet'][0].cur_pos
                    elif 'coin' in obj.get_name():
                        cur_pos = self.env.objs['piggie_bank'][0].cur_pos
                    elif 'cube' in obj.get_name():
                        cur_pos = self.env.objs['cube_cabinet'][0].cur_pos
                else:
                    cur_pos = obj.cur_pos
                obj_states += [*cur_pos]
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        if state_value not in default_states:
                            state = obj.states[state_value].get_value(self.env)
                            obj_states.append(1 if state else 0)
                        else:
                            continue
        obs = [0, 0, 0]  # Agent position (x, y) and direction
        obs += obj_states
        return gym.spaces.Box(low=0, high=max(self.env.width, self.env.height), 
                            shape=(len(obs),), dtype=np.float32)

    def gen_obs(self):
        default_states = [
            'atsamelocation',
            'infovofrobot',
            'inleftreachofrobot',
            'inrightreachofrobot',
            'inside',
            'nextto',
        ]
        obj_states = []
        for obj_type in self.env.objs.values():
            for obj in obj_type:
                # Some objects are attached to others or inside others, and are off the grid in code but should be in same position as parent object
                if obj.cur_pos is None:
                    if 'gear' in obj.get_name():
                        cur_pos = self.env.objs['gear_toy'][0].cur_pos
                    elif 'winnie' in obj.get_name():
                        cur_pos = self.env.objs['winnie_cabinet'][0].cur_pos
                    elif 'coin' in obj.get_name():
                        cur_pos = self.env.objs['piggie_bank'][0].cur_pos
                    elif 'cube' in obj.get_name():
                        cur_pos = self.env.objs['cube_cabinet'][0].cur_pos
                else:
                    cur_pos = obj.cur_pos
                obj_states += [*cur_pos]
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        if state_value not in default_states:
                            state = obj.states[state_value].get_value(self.env)
                            obj_states.append(1 if state else 0)
                        else:
                            continue
        obs = list(self.env.agent_pos) + [self.env.agent_dir] + obj_states
        return np.array(obs, dtype=np.float32)
    

