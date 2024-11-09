from mini_behavior.roomgrid import *
from mini_behavior.register import register
from enum import IntEnum
from gym import spaces


class PickingUpARattleEnv(RoomGrid):
    """
    Environment in which the agent picks up a rattle from the floor
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'rattle': 1}

        self.mission = 'pick up the rattle'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        self.actions = PickingUpARattleEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_objs(self):
        rattle = self.objs['rattle'][0]
        self.place_obj(rattle)

    def _init_conditions(self):
        assert 'rattle' in self.objs.keys(), "No rattle"
        rattle = self.objs['rattle'][0]
        assert rattle.check_abs_state(self, 'onfloor')
        return True

    def _end_conditions(self):
        rattle = self.objs['rattle'][0]
        if not rattle.check_abs_state(self, 'onfloor'):
            return True
        return False


register(
    id='MiniGrid-PickingUpARattle-16x16-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv'
)

register(
    id='MiniGrid-PickingUpARattle-8x8-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={"room_size": 8, "max_steps": 1000}
)

register(
    id='MiniGrid-PickingUpARattle-6x6-N2-v0',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={"room_size": 6, "max_steps": 1000}
)

register(
    id='MiniGrid-PickingUpARattle-16x16-N2-v1',
    entry_point='mini_behavior.envs:PickingUpARattleEnv',
    kwargs={'mode': 'cartesian'}
)
