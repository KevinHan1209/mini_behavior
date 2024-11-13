from mini_behavior.roomgrid import *
from mini_behavior.register import register

class GearToyEnv(RoomGrid):
    '''
    Environment in which agent is instructed to remove all gears from the toy and place them in a bucket
    '''
    def __init__(
            self,
            mode='primitive',
            room_size=8,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
            num_gears = 3,
            exploration_type = None
    ):
        self.num_gears = num_gears
        num_objs = {'gear': num_gears, 'gear_pole': 1, 'bucket': 1}
        self.mission = 'play gear toy'
        self.frames = []

        super().__init__(mode=mode,
                num_objs=num_objs,
                room_size=room_size,
                num_rows=num_rows,
                num_cols=num_cols,
                max_steps=max_steps,
                exploration_type = exploration_type
                )
        
    def _gen_objs(self):
        gear_pole = self.objs['gear_pole'][0]
        bucket = self.objs['bucket'][0]

        self.place_obj(gear_pole)
        self.place_obj(bucket)
        
        for gear in self.objs['gear']:
            self.put_obj(gear, *gear_pole.cur_pos, 0)

    def _init_conditions(self):
        for obj_type in ['gear', 'gear_pole', 'bucket']:
            assert obj_type in self.objs.keys(), f"No {obj_type}"

        gear = self.objs['gear']
        gear_pole = self.objs['gear_pole'][0]
        bucket = self.objs['bucket'][0]

        for gear in self.objs['gear']:
            assert gear.check_rel_state(self, gear_pole, 'inside')
            assert not gear.check_rel_state(self, bucket, 'inside')
        assert gear_pole.check_abs_state(self, 'onfloor')
        assert bucket.check_abs_state(self, 'onfloor')

        return True
    
    def _end_conditions(self):
        bucket_checker = 0
        bucket = self.objs['bucket'][0]

        for gear in self.objs['gear']:
            if gear.check_rel_state(self, bucket, 'inside'):
                bucket_checker += 1

        if bucket_checker != self.num_gears:
            return False
    
        return True
        
# non human input env
register(
    id='MiniGrid-GearToy-16x16-N2-v0',
    entry_point='mini_behavior.envs:GearToyEnv'
)

# human input env
register(
    id='MiniGrid-GearToy-16x16-N2-v1',
    entry_point='mini_behavior.envs:GearToyEnv',
    kwargs={'mode': 'cartesian'}
)