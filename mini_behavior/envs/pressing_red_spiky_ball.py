from mini_behavior.roomgrid import *
from mini_behavior.register import register
from enum import IntEnum
from gym import spaces

class PressingRedSpikyBallEnv(RoomGrid):
    """
    Environment in which the agent interacts with a red spiky ball
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'red_spiky_ball': 1}

        self.mission = 'interact with the red spiky ball'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        self.actions = PressingRedSpikyBallEnv.Actions
        self.action_space is spaces.Discrete(len(self.actions))

    def _gen_objs(self):
        red_spiky_ball = self.objs['red_spiky_ball'][0]
        self.place_obj(red_spiky_ball)

    def _init_conditions(self):
        assert 'red_spiky_ball' in self.objs.keys(), "No Red Spiky Ball"

        red_spiky_ball = self.objs['red_spiky_ball'][0]
        # Initial state: red spiky ball is intact (not deformed)
        assert red_spiky_ball.check_abs_state(self, 'intact')

        return True

    def _end_conditions(self):
        red_spiky_ball = self.objs['red_spiky_ball'][0]
        # End condition: red spiky ball is deformed (pressed)
        if red_spiky_ball.check_abs_state(self, 'deformed'):
            return True
        else:
            return False

register(
    id='MiniGrid-PressingRedSpikyBall-16x16-N2-v0',
    entry_point='mini_behavior.envs:PressingRedSpikyBallEnv'
)

register(
    id='MiniGrid-PressingRedSpikyBall-8x8-N2-v0',
    entry_point='mini_behavior.envs:PressingRedSpikyBallEnv',
    kwargs={"room_size": 8, "max_steps": 1000}
)

register(
    id='MiniGrid-PressingRedSpikyBall-6x6-N2-v0',
    entry_point='mini_behavior.envs:PressingRedSpikyBallEnv',
    kwargs={"room_size": 6, "max_steps": 1000}
)
