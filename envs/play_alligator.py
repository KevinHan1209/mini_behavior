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
    ):
        num_objs = {'mallet': 1, 'music_box': 1}
        self.mission = 'play music box'
        self.frames = []

        super().__init__(mode=mode,
                    num_objs=num_objs,
                    room_size=room_size,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    max_steps=max_steps
                    )
        
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