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
            room_size=8,
            room_width=26,
            room_height=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            agent_pos=(15, 11),
            extrinsic_rewards=None,
            object_positions={
                'gear_toy': (21, 3),
                'stroller': (21, 6),
                'tree_busy_box': (4, 4),
                'piggie_bank': (4, 9),
                'shape_sorter': (4, 13),
                'bucket_toy': (21, 9),
                'cube_cabinet': (19, 1),
                'alligator_busy_box': (18, 1),
                'beach_ball': (14, 5),
                'ring_toy': (17, 11),
                'music_toy': (13, 13),
                'red_spiky_ball': (13, 9),
                'broom_set': (12, 9),
                'farm_toy': (12, 10),
                'cart_toy': (8, 12),
                'broom': (8, 14),
                'winnie_cabinet': (5, 14),
                'rattle': (14, 11)
            }
    ):
        '''
        All unique objects in the dataset: 

        {nan, geartoy, stroller, tree busy box, shapesorter with 3 shapes, red spiky ball, 
        'bc' (black cloth, not implemented), cabinet with winnie, rattle, broom set, broom, 
        farm toy, pink beach ball, pink pig and coins, doorknob (not implemented), music toy alligator,
        alligator busy box, cart, yellow donut ring, bucket w/ 6 balls, cube w/ 6 diff colors}
        '''
        self.num_objs = {
            'shape_toy': 3, 'shape_sorter': 1, 'gear': 6, 'gear_toy': 1, 'stroller': 1,
            'tree_busy_box': 1, 'red_spiky_ball': 1, 'rattle': 1, 'broom_set': 1,
            'broom': 1, 'farm_toy': 1, 'beach_ball': 1, 'piggie_bank': 1, 'mini_broom': 1,
            'coin': 10, 'cube': 6, 'alligator_busy_box': 1, 'music_toy': 1, 'cube_cabinet': 1,
            'cart_toy': 1, 'ring_toy': 1, 'bucket_toy': 1, 'winnie': 1, 'winnie_cabinet': 1
        }

        self.mission = 'explore'
        self.object_positions = object_positions

        super().__init__(mode=mode,
                         test_env=test_env,
                         num_objs=self.num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps,
                         room_width=room_width,
                         room_height=room_height,
                         agent_pos=agent_pos,
                         extrinsic_rewards=extrinsic_rewards
                         )

        self.locomotion_actions = MiniBehaviorEnv.LocoActions
        self.manipulation_actions = MiniBehaviorEnv.ObjectActions
        self.action_space = spaces.MultiDiscrete([
            len(self.manipulation_actions),  # First dimension: left arm actions
            len(self.manipulation_actions),  # Second dimension: right arm actions
            len(self.locomotion_actions)     # Third dimension: locomotion actions
        ])

    def _gen_objs(self):
        """Places all objects dynamically from self.num_objs"""
        for obj_type in self.num_objs.keys():
            for obj in self.objs[obj_type]:
                if "winnie" in obj_type and not "winnie_cabinet" in obj_type:
                    # Begin with winnie inside the cabinet
                    self.objs["winnie_cabinet"][0].states["contains"].add_obj(obj)
                    obj.states["inside"].set_value(True)
                elif "gear" in obj_type and not "gear_toy" in obj_type:
                    self.objs["gear_toy"][0].states["contains"].add_obj(obj)
                    obj.states["attached"].set_value(True)
                    self.objs["gear_toy"][0].states["attached"].set_value(True)
                elif "coin" in obj_type:
                    # Place coins in the piggie bank
                    self.objs["piggie_bank"][0].states["contains"].add_obj(obj)
                    obj.states["inside"].set_value(True)
                elif "cube" in obj_type and not "cube_cabinet" in obj_type:
                    # Place cubes in the cube cabinet
                    self.objs["cube_cabinet"][0].states["contains"].add_obj(obj)
                    obj.states["inside"].set_value(True)
                else:
                    # If position is specified, use it
                    if self.object_positions and obj_type in self.object_positions:
                        pos = self.object_positions[obj_type]
                        self.put_obj(obj, pos[0], pos[1])
                    else:
                        # Otherwise use default placement
                        self.place_obj(obj)

    def _init_conditions(self):
        """Checks all objects are initialized properly"""
        for obj_type in self.num_objs.keys():
            assert obj_type in self.objs, f"No {obj_type}"
            obj = self.objs[obj_type][0]  # Get the first object of this type

    def _end_conditions(self):
        """Task is exploration"""
        return False
    
    def get_room_size(self):
        return self.room_size
    '''
    register(
        id='MiniGrid-MultiToy-8x8-N2-v0',
        entry_point='mini_behavior.envs.multitoy:MultiToyEnv',
    )
'''