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


class InRightHandOfRobot(AbsoluteObjectState):
    # return true if agent is carrying the object
    def get_value(self, env=None):
        return not len(env.carrying['right']) == 0
    
class InLeftHandOfRobot(AbsoluteObjectState):
    # return true if agent is carrying the object
    def get_value(self, env=None):
        return not len(env.carrying['left']) == 0


class InLeftReachOfRobot(AbsoluteObjectState):
    # return true if obj is reachable by agent
    def get_value(self, env):
        # obj not reachable if inside closed obj2
        inside = self.obj.inside_of
        if inside is not None and 'openable' in inside.states.keys() and not inside.check_abs_state(env, 'openable'):
            return False

        carrying = self.obj.check_abs_state(env, 'inlefthandofrobot')

        if self.obj.is_furniture():
            in_front = False
            for pos in self.obj.all_pos:
                if np.all(pos == env.front_pos):
                    in_front = True
                    break
        else:
            in_position = any(np.all(self.obj.cur_pos == pos) for pos in [env.front_pos, env.upper_left_pos, env.left_pos])
        return carrying or in_position
    
class InRightReachOfRobot(AbsoluteObjectState):
    # return true if obj is reachable by agent
    def get_value(self, env):
        # obj not reachable if inside closed obj2
        inside = self.obj.inside_of
        if inside is not None and 'openable' in inside.states.keys() and not inside.check_abs_state(env, 'openable'):
            return False

        carrying = self.obj.check_abs_state(env, 'inrighthandofrobot')

        if self.obj.is_furniture():
            in_front = False
            for pos in self.obj.all_pos:
                if np.all(pos == env.front_pos):
                    in_front = True
                    break
        else:
            in_position = any(np.all(self.obj.cur_pos == pos) for pos in [env.front_pos, env.upper_right_pos, env.right_pos])
        return carrying or in_position


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
    '''
    Dependent on Contains state. Logic implemented in assemble/disassemble action class. 
    '''
    def __init__(self, obj, key):
        super(Attached, self).__init__(obj, key)
    
    def get_value(self, env):
        # Logic implemented in Contains
        return self.value


class Contains(AbilityState):
    '''
    For objects which can contain other objects
    '''
    def __init__(self, obj, key):
        super(Contains, self).__init__(obj, key)
        self.contained_objects = []

    def add_obj(self, obj):
        self.contained_objects.append(obj)
    
    def remove_obj(self):
        return self.contained_objects.pop(0)
    
    def get_num_objs(self):
        return len(self.contained_objects)
    
    def get_value(self, env):
        return len(self.contained_objects) != 0
    
    def get_contained_objs(self):
        return self.contained_objects

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
    
class Inside(AbilityState):
    """
    Depends on contains state. Logic implemented in dropin/takeout action class.
    """
    def __init__(self, obj, key):
        super(Inside, self).__init__(obj, key)

    def get_value(self, env):
        return self.value
    
class Kicked(AbilityState):
    def __init__(self, obj, key):
        super(Kicked, self).__init__(obj, key)

    def get_value(self, env):
        return self.value

class Noise(AbilityState):
    def __init__(self, obj, key):
        super(Noise, self).__init__(obj, key)

    def get_value(self, env):
        """
        True depending on the action performed and object
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

class Thrown(AbilityState):
    def __init__(self, obj, key):
        super(Thrown, self).__init__(obj, key)

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

