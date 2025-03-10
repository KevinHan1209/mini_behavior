from mini_behavior.roomgrid import *
from mini_behavior.register import register
from enum import IntEnum
from gym import spaces


class MultiToyEnv(RoomGrid):
    """
    Environment in which the agent explores a multitude of toys
    """

    def __init__(
            self,
            test_env=False,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        self.num_objs = {
            'alligator_busy_box': 1, 'beach_ball': 1, 'cart_toy': 1, 'coin': 1, 
            'music_toy': 1, 'piggie_bank': 1, 'gear_toy': 1, 'gear' : 3,
            'rattle': 1
        }

        self.mission = 'explore'

        super().__init__(mode=mode,
                         test_env=test_env,
                         num_objs=self.num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

        self.actions = MultiToyEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def _gen_objs(self):
        """Places all objects dynamically from self.num_objs"""
        for obj_type in self.num_objs.keys():
            for obj in self.objs[obj_type]:
                self.place_obj(obj)

    def _init_conditions(self):
        """Checks all objects are initialized properly"""
        for obj_type in self.num_objs.keys():
            assert obj_type in self.objs, f"No {obj_type}"
            obj = self.objs[obj_type][0]  # Get the first object of this type
            assert obj.check_abs_state(self, 'onfloor')

    def _end_conditions(self):
        """Task is exploration"""
        pass
