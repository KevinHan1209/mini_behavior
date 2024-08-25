from mini_behavior.roomgrid import *
from mini_behavior.register import register
from enum import IntEnum
from gym import spaces


class ToyInteractionEnv(RoomGrid):
    """
    Environment in which the agent interacts with a Rattle and a Red Spiky Ball
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'rattle': 1, 'red_spiky_ball': 1}

        self.mission = 'interact with the Rattle and Red Spiky Ball'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        self.actions = ToyInteractionEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_objs(self):
        rattle = self.objs['rattle'][0]
        red_spiky_ball = self.objs['red_spiky_ball'][0]

        self.place_obj(rattle)
        self.place_obj(red_spiky_ball)

    def _init_conditions(self):
        for obj_type in ['rattle', 'red_spiky_ball']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        rattle = self.objs['rattle'][0]
        red_spiky_ball = self.objs['red_spiky_ball'][0]

        # Initial state: Rattle is silent, Red Spiky Ball is not deformed
        assert rattle.check_abs_state(self, 'silent')
        assert red_spiky_ball.check_abs_state(self, 'intact')

        return True

    def _end_conditions(self):
        rattle = self.objs['rattle'][0]
        red_spiky_ball = self.objs['red_spiky_ball'][0]

        # End condition: Rattle is making noise (shaken), Red Spiky Ball is deformed (pressed)
        if rattle.check_abs_state(self, 'noise') and red_spiky_ball.check_abs_state(self, 'deformed'):
            return True
        else:
            return False


register(
    id='MiniGrid-ToyInteraction-16x16-N2-v0',
    entry_point='mini_behavior.envs:ToyInteractionEnv'
)

register(
    id='MiniGrid-ToyInteraction-8x8-N2-v0',
    entry_point='mini_behavior.envs:ToyInteractionEnv',
    kwargs={"room_size": 8, "max_steps": 1000}
)

register(
    id='MiniGrid-ToyInteraction-6x6-N2-v0',
    entry_point='mini_behavior.envs:ToyInteractionEnv',
    kwargs={"room_size": 6, "max_steps": 1000}
)

register(
    id='MiniGrid-ToyInteraction-16x16-N2-v1',
    entry_point='mini_behavior.envs:ToyInteractionEnv',
    kwargs={'mode': 'cartesian'}
)
