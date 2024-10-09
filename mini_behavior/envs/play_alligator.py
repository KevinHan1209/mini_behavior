from mini_behavior.roomgrid import *
from mini_behavior.register import register

from array2gif import write_gif

class PlayAlligatorEnv(RoomGrid):
    '''
    Environment in which agent is instructed to play the music alligator toy using a mallet
    '''
    def __init__(
            self,
            mode='primitive',
            room_size=8,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            exploration_type = None
    ):
        num_objs = {'mallet': 1, 'music_box': 1}
        self.mission = 'play music box'
        self.frames = []


        super().__init__(mode=mode,
                    num_objs=num_objs,
                    room_size=room_size,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    max_steps=max_steps,
                    exploration_type = exploration_type
                    )
        '''
        for obj_type1, obj_type2 in zip(self.objs.values(), self.objs.values()):
            for obj1, obj2 in zip(obj_type1, obj_type2):
                for state_value in obj1.states:
                    print(state_value)
                    if isinstance(obj1.states[state_value], RelativeObjectState):
                        continue
                    print(obj1.states[state_value].get_value(self))
        '''

    '''
    def gen_APT_obs(self):
        # Generates all object states as well as agent's current position and direction
        obj_states = []
        for obj_type in self.objs.values():
            for obj in obj_type:
                for state_value in obj.states:
                    if isinstance(obj.states[state_value], RelativeObjectState):
                        continue
                    state = obj.states[state_value].get_value(self)
                    if state == True:
                        obj_states.append(1)
                    else:
                        obj_states.append(0)
        obs = []
        obs.append(self.agent_pos[0])
        obs.append(self.agent_dir)
        obs += obj_states
        return obs
        '''
        
    def _gen_objs(self):
        mallet = self.objs['mallet'][0]
        music_box = self.objs['music_box'][0]
        self.place_obj(mallet)
        self.place_obj(music_box)
        music_box.states['playable'].set_value(False)


    def _init_conditions(self):
        for obj_type in ['mallet', 'music_box']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        mallet = self.objs['mallet'][0]
        music_box = self.objs['music_box'][0]

        assert mallet.check_abs_state(self, 'onfloor')
        assert music_box.check_abs_state(self, 'onfloor')
        assert not music_box.check_abs_state(self, 'playable')

        return True

    def _end_conditions(self):
        for music_box in self.objs['music_box']:
            if not music_box.check_abs_state(self, 'playable'):
                return False
        return True
    
    '''
    def step(self):
        super().step()
        self.frames.append(np.moveaxis(super().render("rgb_array"), 2, 0))
    
    def reset(self):
        super().reset()
        gif_name = f"test_visualization"
        if self.frames != []:
            write_gif(np.array(self.frames),
                    gif_name,
                    fps=1 / .25)
        self.frames = []
    '''

    
 # non human input env
register(
    id='MiniGrid-PlayAlligator-16x16-N2-v0',
    entry_point='mini_behavior.envs:PlayAlligatorEnv'
)

# human input env
register(
    id='MiniGrid-PlayAlligator-16x16-N2-v1',
    entry_point='mini_behavior.envs:PlayAlligatorEnv',
    kwargs={'mode': 'cartesian'}
)