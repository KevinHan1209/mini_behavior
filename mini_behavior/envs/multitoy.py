from mini_behavior.roomgrid import *
from mini_behavior.register import register
from enum import IntEnum
from gym import spaces


class MultiToyEnv(RoomGrid):
    """
    Environment in which the agent picks up a rattle from the floor
    """

    def __init__(
            self,
            mode='primitive',
            room_size=32,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'alligator_busy_box': 1, 'ball': 1, 'beach_ball': 1, 'music_toy': 1, 'farm_toy': 1, 'piggie_bank': 1, 'rattle': 1, 'red_spiky_ball': 1, 'winnie_cabinet': 1}

        self.mission = 'explore'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        self.actions = MultiToyEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_objs(self):
        alligator_busy_box = self.objs['alligator_busy_box'][0]
        ball = self.objs['ball'][0]
        beach_ball = self.objs['beach_ball'][0]
        music_toy = self.objs['music_toy'][0]
        farm_toy = self.objs['farm_toy'][0]
        rattle = self.objs['rattle'][0]
        red_spiky_ball = self.objs['red_spiky_ball'][0]
        piggie_bank = self.objs['piggie_bank'][0]
        winnie_cabinet = self.objs['winnie_cabinet'][0]

        self.place_obj(alligator_busy_box)
        self.place_obj(ball)
        self.place_obj(beach_ball)
        self.place_obj(music_toy)
        self.place_obj(farm_toy)
        self.place_obj(rattle)
        self.place_obj(red_spiky_ball)
        self.place_obj(piggie_bank)
        self.place_obj(winnie_cabinet)


    def _init_conditions(self):
        for obj_type in ['alligator_busy_box', 'ball', 'beach_ball', 'music_toy', 'farm_toy', 'rattle', 'red_spiky_ball', 'piggie_bank', 'winnie_cabinet']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        alligator_busy_box = self.objs['alligator_busy_box'][0]
        ball = self.objs['ball'][0]
        beach_ball = self.objs['beach_ball'][0]
        music_toy = self.objs['music_toy'][0]
        farm_toy = self.objs['farm_toy'][0]
        rattle = self.objs['rattle'][0]
        red_spiky_ball = self.objs['red_spiky_ball'][0]
        piggie_bank = self.objs['piggie_bank'][0]
        winnie_cabinet = self.objs['winnie_cabinet'][0]

        assert alligator_busy_box.check_abs_state(self, 'onfloor')
        assert ball.check_abs_state(self, 'onfloor')
        assert beach_ball.check_abs_state(self, 'onfloor')
        assert music_toy.check_abs_state(self, 'onfloor')
        assert farm_toy.check_abs_state(self, 'onfloor')
        assert rattle.check_abs_state(self, 'onfloor')
        assert red_spiky_ball.check_abs_state(self, 'onfloor')
        assert piggie_bank.check_abs_state(self, 'onfloor')
        assert winnie_cabinet.check_abs_state(self, 'onfloor')



    # def _end_conditions(self):
    #     rattle = self.objs['rattle'][0]
    #     if not rattle.check_abs_state(self, 'onfloor'):
    #         return True
    #     return False

    def _end_conditions(self):
        return False

'''
register(
    id='MiniGrid-MultiToy-32x32-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv'
)

register(
    id='MiniGrid-MultiToy-8x8-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={"room_size": 8, "max_steps": 1000}
)

register(
    id='MiniGrid-MultiToy-6x6-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={"room_size": 6, "max_steps": 1000}
)

register(
    id='MiniGrid-MultiToy-16x16-N2-v1',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={'mode': 'cartesian'}
)
'''