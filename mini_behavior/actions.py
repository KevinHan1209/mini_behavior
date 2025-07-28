import numpy as np

def find_tool(env, possible_tool_types, arm):
    # returns whether agent is carrying a obj of possible_tool_types, and the obj_instance
    for tool_type in possible_tool_types:
        tools = env.objs.get(tool_type, []) # objs of type tool in the env
        for tool in tools:
            if tool.check_abs_state(env, 'in' + arm + 'handofrobot'):
                return True
    return False

class BaseAction:
    def __init__(self, env):
        """
        initialize action
        """
        super(BaseAction, self).__init__()
        self.env = env
        self.key = None

    def can(self, obj, arm):
        """
        check if possible to do action
        """

        # check if possible to do the action on the object
        if not obj.possible_action(self.key):
            return False

        # check if the object is in reach of the agent
        if not obj.check_abs_state(self.env, 'in' + arm + 'reachofrobot'):
            return False

        return True

    def do(self, obj, arm):
        """
        do action
        """
        assert self.can(obj, arm), 'Cannot perform action'

class Assemble(BaseAction):
    def __init__(self, env):
        super(Assemble, self).__init__(env)
        self.key = 'assemble'
        self.tools = ["broom", "gear"]

    def can(self, obj, arm):
        """
        can only do this if 
        -agent is holding the correct objects
        -the existing state is Attached = False
        -contain objecct is not at capacity
        """
        opp_arm = "left" if arm == "right" else "right"
        if self.env.carrying[arm] == set() or self.env.carrying[opp_arm] == set():
            return False
        # Check that both arms are holding the right objects
        if super().can(obj, arm):
            if "gear_toy" in obj.get_name(): 
                if not "gear" in list(self.env.carrying[opp_arm])[0].get_name():
                    return False
            elif "gear" in obj.get_name():
                if not "gear_toy" in list(self.env.carrying[opp_arm])[0].get_name():
                    return False
            elif "broom_set" in obj.get_name():
                if not "mini_broom" in list(self.env.carrying[opp_arm])[0].get_name():
                    return False
            elif "mini_broom" in obj.get_name():
                if not "broom_set" in list(self.env.carrying[opp_arm])[0].get_name():
                    return False
        else:
            return False

        # check if container object is at capacity
        main_arm = arm if ("gear_toy" in obj.get_name() or "broom_set" in obj.get_name()) else opp_arm
        container_obj = list(self.env.carrying[main_arm])[0]
        if container_obj.states['contains'].get_num_objs() > container_obj.max_contain:
            return False

        return True

    def do(self, obj, arm):
        """
        Find arm with the "attachee" (gear pole or broom set) and attach to that arm
        """
        super().do(obj, arm)
        obj.states['attached'].set_value(True)
        opp_arm = "left" if arm == "right" else "right"
        main_arm = arm if ("gear_toy" in obj.get_name() or "broom_set" in obj.get_name()) else opp_arm
        other_arm = "left" if main_arm == "right" else "right"
        list(self.env.carrying[other_arm])[0].states['attached'].set_value(True)
        list(self.env.carrying[main_arm])[0].states["attached"].set_value(True)
        list(self.env.carrying[main_arm])[0].states["contains"].add_obj(list(self.env.carrying[other_arm])[0])
        self.env.carrying[other_arm].discard(obj)
        
class Brush(BaseAction):
    def __init__(self, env):
        super(Brush, self).__init__(env)
        self.key = 'brush'

    def can(self, obj, arm):
        """
        can only do this if:
        - agent is holding the correct objects (any type of broom)
        """
        if not super().can(obj, arm):
            return False
        if "broom" not in obj.get_name() or "broom_set" in obj.get_name():
            return False
        
        return True

    def do(self, obj, arm):
        super().do(obj, arm)
        obj.states['usebrush'].set_value(True)

class Disassemble(BaseAction):
    """
    Disassembles two objects. Ends with the object that was inside in agent's hand.
    """
    def __init__(self, env):
        super(Disassemble, self).__init__(env)
        self.key = 'disassemble'
    
    def can(self, obj, arm):
        """
        can only do this if Attached is True
        """
        if not super().can(obj, arm) or not obj.states['attached'].get_value(self.env):
            return False
        assert "gear_toy" in obj.get_name() or "broom_set" in obj.get_name()
        opp_arm = "left" if arm == "right" else "right"
        
        return not self.env.carrying[opp_arm]

    def do(self, obj, arm):
        print(f"\n[DEBUG Disassemble] Attempting to disassemble:")
        print(f"  Object: {obj.name if hasattr(obj, 'name') else obj}")
        print(f"  Arm: {arm}")
        print(f"  Contains state exists: {'contains' in obj.states if hasattr(obj, 'states') else 'No states'}")
        if hasattr(obj, 'states') and 'contains' in obj.states:
            contained = obj.states['contains'].get_contained_objs()
            print(f"  Contained objects: {[o.name for o in contained] if contained else 'empty'}")
            print(f"  Number of contained objects: {obj.states['contains'].get_num_objs()}")
        
        super().do(obj, arm)
        other_arm = "left" if arm == "right" else "right"
        detached_object = obj.states["contains"].remove_obj()
        detached_object.states["attached"].set_value(False)
        self.env.carrying[other_arm].add(detached_object)
        
        # If container has no more objects, set its attached state to false
        if obj.states["contains"].get_num_objs() == 0:
            obj.states["attached"].set_value(False)


class Close(BaseAction):
    def __init__(self, env):
        super(Close, self).__init__(env)
        self.key = 'close'

    def do(self, obj):
        super().do(obj)
        obj.states['open'].set_value(False)
        obj.update(self.env)


class Drop(BaseAction):
    def __init__(self, env):
        super(Drop, self).__init__(env)
        self.key = 'drop'

    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if furniture is None and obj is None:
                if last_furniture is not None or last_obj is not None:
                    dims.append(i)

            last_furniture = furniture
            last_obj = obj

        return dims

    def can(self, obj, arm, pos):
        """
        can drop obj if:
        - agent is carrying obj
        - there is no obj in base of forward cell
        """
        if not super().can(obj, arm):
            return False

        if not obj.check_abs_state(self.env, 'in' + arm + 'handofrobot'):
            return False

        #fwd_pos = self.env.front_pos
        dims = self.drop_dims(pos)
        obj.available_dims = dims
        return dims != []

    def do(self, obj, dim, arm, pos):
        assert self.can(obj, arm, pos)

        self.env.carrying[arm].discard(obj)

        #fwd_pos = self.env.front_pos

        # change object properties
        obj.cur_pos = pos
        # change agent / grid
        self.env.grid.set(*pos, obj, dim)


class DropIn(BaseAction):
    def __init__(self, env):
        super(DropIn, self).__init__(env)
        self.key = 'dropin'

    def can(self, obj, arm):
        """
        obj references the container object
        can dropin if:
        - agent is carrying another obj
        - object to be dropped in is not at capacity
        """
        if not super().can(obj, arm):
            return False

        if not self.env.carrying[arm]:
            return False

        if obj.states['contains'].get_num_objs() > obj.max_contain:
            return False
        
        # Edge cases
        if "piggie_bank" in obj.get_name():
            # Cannot drop in if object is not a coin or piggie bank is not open
            return "coin" in list(self.env.carrying[arm])[0].get_name() and obj.states['open'].get_value(self.env)

        if "shape_sorter" in obj.get_name():
            # Cannot drop in if object is not a shape
            return "shape_toy" in list(self.env.carrying[arm])[0].get_name()

        return True

    def do(self, obj, arm):
        # dropin
        super().do(obj, arm)
        list(self.env.carrying[arm])[0].states["inside"].set_value(True)
        obj.states['contains'].add_obj(list(self.env.carrying[arm])[0])
        self.env.carrying[arm].clear()
        


class HitWithObject(BaseAction):
    def __init__(self, env):
        super(HitWithObject, self).__init__(env)
        self.key = 'hitwithobject'

    def can(self, obj, arm):
        """
        agent can hit anything with anything it's carrying
        """
        return super().can(obj, arm) and self.env.carrying[arm]

    def do(self, obj, arm):
        list(self.env.carrying[arm])[0].states["hitter"].set_value(True)
        obj.states["gothit"].set_value(True)

class Hit(BaseAction):
    def __init__(self, env):
        super(Hit, self).__init__(env)
        self.key = 'hit'

    def can(self, obj, arm):
        """
        agent can hit anything with arms. just make sure agent is not carrying or else that should be HitWithObject
        """
        if self.env.carrying[arm]:
            return False
        return super().can(obj, arm)

    def do(self, obj, arm):
        super().do(obj, arm)
        obj.states["gothit"].set_value(True)

class Mouthing(BaseAction):
    def __init__(self, env):
        super(Mouthing, self).__init__(env)
        self.key = 'mouthing'
    def can(self, obj, arm):
        """
        agent can mouth anything it's carrying
        """
        return super().can(obj, arm) and self.env.carrying[arm]
    def do(self, obj, arm):
        obj.states['mouthed'].set_value(True)

class TakeOut(BaseAction):
    def __init__(self, env):
        super(TakeOut, self).__init__(env)
        self.key = 'take_out'

    def can(self, obj, arm):
        """
        can takeout obj if:
        - container obj actually contains obj
        - agent not holding anything else
        """
        if not super().can(obj, arm):
            return False
        if not obj.states['contains'].get_value(self.env):
            return False

        if len(self.env.carrying[arm]) != 0:
            return False
        
        # Edge cases
        if "piggie_bank" in obj.get_name():
            # Cannot take out piggie bank is not open
            return obj.states['open'].get_value(self.env)
        
        return True
    
    def do(self, obj, arm):
        super().do(obj, arm)
        takeout_object = obj.states['contains'].remove_obj()
        takeout_object.states["inside"].set_value(False)
        self.env.carrying[arm].add(takeout_object)



class Open(BaseAction):
    def __init__(self, env):
        super(Open, self).__init__(env)
        self.key = 'open'

    def do(self, obj):
        super().do(obj)
        obj.states['openable'].set_value(True)
        obj.update(self.env)


class Pickup(BaseAction):
    def __init__(self, env):
        super(Pickup, self).__init__(env)
        self.key = 'pickup'

    def can(self, obj, arm):
        if not super().can(obj, arm):
            return False

        # For primitive action type, can only carry one object at a time
        if len(self.env.carrying[arm]) != 0 and self.env.mode == "primitive":
            assert len(self.env.carrying[arm]) == 1
            return False

        # cannot pickup if carrying
        if obj.check_abs_state(self.env, 'in' + arm + 'handofrobot'):
            return False

        return True

    def do(self, obj, arm):
        super().do(obj, arm)
        self.env.carrying[arm].add(obj)

        objs = self.env.grid.get_all_objs(*obj.cur_pos)
        dim = objs.index(obj)

        # remove obj from the grid and shift remaining objs
        self.env.grid.remove(*obj.cur_pos, obj)

        if dim < 2:
            new_objs = objs[: dim] + objs[dim + 1:] + [None]
            assert len(new_objs) == 3
            self.env.grid.set_all_objs(*obj.cur_pos, new_objs)

        # update cur_pos of obj
        obj.update_pos(np.array([-1, -1]))


        # check dependencies
        assert obj.check_abs_state(self.env, 'in' + arm + 'handofrobot')
        assert not obj.check_abs_state(self.env, 'onfloor')

class Pull(BaseAction):
    def __init__(self, env):
        super(Pull, self).__init__(env)
        self.key = 'pull'

    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if furniture is None and obj is None:
                if last_furniture is not None or last_obj is not None:
                    dims.append(i)

            last_furniture = furniture
            last_obj = obj

        return dims

    def can(self, obj, arm):
        if not super().can(obj, arm):
            return False
        # Agent cannot be carrying another item in the same arm when pulling the cart
        if len(self.env.carrying[arm]) != 0:
            assert len(self.env.carrying[arm]) == 1
            return False
        
        agent_pos = list(self.env.agent_pos).copy()
        back_pos = agent_pos - self.env.dir_vec
        dims = self.drop_dims(back_pos)
        return int(0) in dims
        
    def do(self, obj, arm):
        agent_old_pos = list(self.env.agent_pos).copy()
        self.env.grid.remove(*obj.cur_pos, obj)
        self.env.grid.set_all_objs(*obj.cur_pos, [None, None, None])
        self.env.agent_pos = agent_old_pos - self.env.dir_vec
        obj.cur_pos = agent_old_pos
        self.env.grid.set(*agent_old_pos, obj, int(0))
        obj.states['pullshed'].set_value(True)

class Push(BaseAction):
    def __init__(self, env):
        super(Push, self).__init__(env)
        self.key = 'push'

    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if furniture is None and obj is None:
                if last_furniture is not None or last_obj is not None:
                    dims.append(i)

            last_furniture = furniture
            last_obj = obj

        return dims

    def can(self, obj, arm):
        if not super().can(obj, arm):
            return False
        # Agent cannot be carrying another item in the same arm when pushing the cart
        if len(self.env.carrying[arm]) != 0:
            assert len(self.env.carrying[arm]) == 1
            return False
        obj_pos = list(obj.cur_pos).copy()
        front_pos = obj_pos + self.env.dir_vec # position in front of cart/stroller
        dims = self.drop_dims(front_pos) # check if that position is filled with an object
        return int(0) in dims

    def do(self, obj, arm):
        obj_pos = list(obj.cur_pos).copy()
        self.env.grid.remove(*obj.cur_pos, obj)
        self.env.grid.set_all_objs(*obj.cur_pos, [None, None, None])
        front_pos = obj_pos + self.env.dir_vec
        obj.cur_pos = front_pos
        self.env.agent_pos = obj_pos
        self.env.grid.set(*front_pos, obj, int(0))
        obj.states['pullshed'].set_value(True)


class NoiseToggle(BaseAction):
    def __init__(self, env):
        super(NoiseToggle, self).__init__(env)
        self.key =  'noise_toggle'

    def can(self, obj, arm):
        return super().can(obj, arm)
        
    def do(self, obj, arm):
        super().do(obj, arm)
        obj.states['noise'].set_value(True)
        

class Throw(BaseAction):
    def __init__(self, env):
        super(Throw, self).__init__(env)
        self.key = 'throw'
        
    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if furniture is None and obj is None:
                if last_furniture is not None or last_obj is not None:
                    dims.append(i)

            last_furniture = furniture
            last_obj = obj

        return dims
    
    def can(self, obj, arm, pos):
        if not super().can(obj, arm):
            return False

        # object must be in hand to throw
        if not obj.check_abs_state(self.env, 'in' + arm + 'handofrobot'):
            return False
        
        throw_pos = pos + self.env.dir_vec # check if position in front of front position is also open
        check_pos = pos if 0 in throw_pos or throw_pos[0] == self.env.width - 1 or throw_pos[1] == self.env.height - 1 else throw_pos # boundary condition
        if check_pos[0] < 0 or check_pos[1] < 0 or check_pos[0] >= self.env.width or check_pos[1] >= self.env.height:
            return False
        dims = self.drop_dims(check_pos)
        return int(0) in dims # Resort to dim = 0 since dims are not really used anymore
    
    def do(self, obj, dim, arm, pos):
        assert self.can(obj, arm, pos)

        self.env.carrying[arm].discard(obj)

        two_fwd_pos = pos + self.env.dir_vec
        throw_pos = pos if 0 in two_fwd_pos or two_fwd_pos[0] == self.env.width - 1 or two_fwd_pos[1] == self.env.height - 1 else two_fwd_pos # boundary condition
        # change object properties
        obj.cur_pos = throw_pos
        # change agent / grid
        self.env.grid.set(*throw_pos, obj, dim)
        obj.states['thrown'].set_value(True)

class Toggle(BaseAction):
    def __init__(self, env):
        super(Toggle, self).__init__(env)
        self.key = 'toggle'

    def do(self, obj, arm):
        """
        toggle from on to off, or off to on
        """
        super().do(obj, arm)
        if any(substring in obj.get_name() for substring in ["winnie_cabinet", "piggie_bank"]):
            open_cur = obj.check_abs_state(self.env, 'open')
            obj.states['open'].set_value(not open_cur)
        elif any(substring in obj.get_name() for substring in ["farm_toy"]):
            popup_cur = obj.check_abs_state(self.env, 'popup')
            obj.states['popup'].set_value(not popup_cur)
        else:
            cur = obj.check_abs_state(self.env, 'toggled')
            obj.states['toggled'].set_value(not cur)

        


