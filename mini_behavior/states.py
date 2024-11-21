from .utils.states_base import *
import numpy as np


def get_obj_cell(self, env):
    obj = self.obj
    cell = [obj_ for obj_ in env.grid.get_all_items(*obj.cur_pos)]
    return obj, cell


###########################################################################################################
# ROBOT RELATED STATES

# TODO: check that .in_view works correctly
class InFOVOfRobot(AbsoluteObjectState):
    # return true if obj is in front of agent
    def get_value(self, env):
        return env.in_view(*self.obj.cur_pos)


class InHandOfRobot(AbsoluteObjectState):
    # return true if agent is carrying the object
    def get_value(self, env=None):
        return np.all(self.obj.cur_pos == np.array([-1, -1]))


class InReachOfRobot(AbsoluteObjectState):
    # return true if obj is reachable by agent
    def get_value(self, env):
        # obj not reachable if inside closed obj2
        inside = self.obj.inside_of
        if inside is not None and 'openable' in inside.states.keys() and not inside.check_abs_state(env, 'openable'):
            return False

        carrying = self.obj.check_abs_state(env, 'inhandofrobot')

        if self.obj.is_furniture():
            in_front = False
            for pos in self.obj.all_pos:
                if np.all(pos == env.front_pos):
                    in_front = True
                    break
        else:
            in_front = np.all(self.obj.cur_pos == env.front_pos)

        return carrying or in_front


class InSameRoomAsRobot(AbsoluteObjectState):
    # return true if agent is in same room as the object
    def get_value(self, env):
        if self.obj.check_abs_state(env, 'inhandofrobot'):
            return True

        obj_room = env.room_from_pos(*self.obj.cur_pos)
        agent_room = env.room_from_pos(*env.agent_pos)
        return np.all(obj_room == agent_room)


###########################################################################################################
# ABSOLUTE OBJECT STATES

class Attached(AbilityState):
    def __init__(self, obj, key):
        super(Attached, self).__init__(obj, key)

    def _update(self, env):
        '''
        Attached combines two or more objects into one. 
        Used for actions which require more sophistication than just dropping an object into another.

        Applies for broom with broom_set, gear with gear_toy, and winnie with winnie_cabinet.

        True if Assemble action is performed
        False if Disassemble action is performed
        '''
        return self.value

class Flipped(AbilityState):
    def __init__(self, obj, key):
        super(Flipped, self).__init__(obj, key)

    def get_value(self, env):
        '''
        NON-BINARY STATE
        Flipped can take on a set of values depending on how many ways you can flip the object
        Done through flip action
        '''
        return self.value

class Noise(AbilityState):
    def __init__(self, obj, key):
        super(Noise, self).__init__(obj, key)

    def get_value(self, env):
        """
        True depending on the action performed and object
        TODO: If True, False if no other action is performed which could induce noise. Probably do this in environment step?
        """
        return self.value
    
class Popup(AbilityState):
    def __init__(self, obj, key):
        super(Popup, self).__init__(obj, key)

    def get_value(self, env):
        """
        True if toggle action is performed on certain objects
        False if toggle action is performed again
        Will duplicate with ToggledOn state
        """
        return self.value

class Opened(AbilityState):
    def __init__(self, obj, key):
        """
        Value changes only when Open action is done on obj
        """
        super(Opened, self).__init__(obj, key)

    def get_value(self, env):
        return self.value

class ToggledOn(AbilityState):
    def __init__(self, obj, key): # env
        super(ToggledOn, self).__init__(obj, key)

    def get_value(self, env):
        return self.value


###########################################################################################################
# RELATIVE OBJECT STATES

class AtSameLocation(RelativeObjectState):
    # returns true if obj is at the same location as other
    # def _update(self, other, env):
    def _get_value(self, other, env=None):
        if other is None:
            return False

        obj_pos = self.obj.all_pos if self.obj.is_furniture() else [self.obj.cur_pos]
        other_pos = other.all_pos if other.is_furniture() else [other.cur_pos]

        for pos_1 in obj_pos:
            for pos_2 in other_pos:
                if np.all(pos_1 == pos_2):
                    return True

        return False


class Inside(RelativeObjectState):
    """
    Inside(obj1, obj2) change ONLY IF Pickup(obj1) or Drop(obj1) is called
    """
    def __init__(self, obj, key): # env
        super(RelativeObjectState, self).__init__(obj, key)
        self.type = 'relative'

    def _get_value(self, other, env=None):
        # return other in self.inside_of
        if self.obj == other or other is None:
            return False

        return other == self.obj.inside_of

    def _set_value(self, other, new_value):
        if new_value:
            self.obj.inside_of = other
            other.contains = self.obj
        else:
            self.obj.inside_of = None
            other.contains = None


# TODO: fix for furniture
class NextTo(RelativeObjectState):
    # return true if objs are next to each other
    def _get_value(self, other, env=None):
        if other is None or self.obj == other:
            return False

        left_1, bottom_1 = self.obj.cur_pos
        right_1 = left_1 + self.obj.width - 1
        top_1 = bottom_1 + self.obj.height - 1

        left_2, bottom_2 = other.cur_pos
        right_2 = left_2 + other.width - 1
        top_2 = bottom_2 + other.height - 1

        # above, below
        if left_1 <= right_2 and left_2 <= right_1:
            if bottom_2 - top_1 == 1 or bottom_1 - top_2 == 1:
                return True

        # left, right
        if top_1 >= bottom_2 and top_2 >= bottom_1:
            if left_1 - right_2 == 1 or left_2 - right_1 == 1:
                return True

        return False


# TODO: fix for 3D
class OnFloor(AbsoluteObjectState):
    def _update(self, env=None):
        if self.obj.check_abs_state(env, 'inhandofrobot'):
            self.value = False
        else:
            self.value = True


class OnTop(RelativeObjectState):
    def __init__(self, obj, key):
        super(OnTop, self).__init__(obj, key)

    def _get_value(self, other, env=None):
        if other is None or self.obj == other:
            return False

        if self.obj.check_abs_state(self, 'inhandofrobot'):
            return False

        obj, cell = get_obj_cell(self, env)
        cell.reverse()
        obj_idx = cell.index(obj)

        if other not in cell:
            return False

        other_idx = cell.index(other)

        if obj_idx >= 0 and other_idx >= 0:
            return obj_idx < other_idx

        return False


class Under(RelativeObjectState):
    def __init__(self, obj, key):
        super(Under, self).__init__(obj, key)

    def _get_value(self, other, env=None):
        if other is None or self.obj == other:
            return False

        obj, cell = get_obj_cell(self, env)

        obj_idx = cell.index(obj)
        other_idx = cell.index(other)

        if obj_idx > 0 and other_idx > 0:
            return obj_idx < other_idx

        return False

