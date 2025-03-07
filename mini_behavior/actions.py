import numpy as np

def find_tool(env, possible_tool_types):
    # returns whether agent is carrying a obj of possible_tool_types, and the obj_instance
    for tool_type in possible_tool_types:
        tools = env.objs.get(tool_type, []) # objs of type tool in the env
        for tool in tools:
            if tool.check_abs_state(env, 'inhandofrobot'):
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

    def can(self, obj):
        """
        can only do this if 
        -baby is holding the correct object, 
        -the existing state is Attached = False, 
        """
        if super().can(obj) and obj.states['attached'].get_value() == False:
            if obj.get_name() == "gear_toy" and find_tool(self.env, ["gear"]):
                return True
            if obj.get_name() == "broom_set" and find_tool(self.env, ["mini_broom"]):
                return True
        return False

    def do(self, obj):
        super().do(obj)
        obj.states['attached'].set_value(True)
        # find correct tool that is in hand of the agent
        if obj.get_name() == "gear_toy":
            toys = self.env.objs.get("gear", []) 
        elif obj.get_name() == "broom_set":
            toys = self.env.objs.get("mini_broom, []")
        for toy in toys:
            if toy.check_abs_state(self.env, 'inhandofrobot'):
                # once found, put tool "inside" of obj
                toy.states['inside'].set_value(obj, True)
                self.env.carrying.discard(obj)
                fwd_pos = self.env.front_pos
                obj.cur_pos = fwd_pos

        fwd_pos = self.env.front_pos
        obj.cur_pos = fwd_pos
        self.env.grid.set(*fwd_pos, obj)

class Disassemble(BaseAction):
    """
    Disassembles two objects. Ends with the object that was inside in agent's hand.
    """
    def __init__(self, env):
        super(Disassemble, self).__init__(env)
        self.key = 'disassemble'
    
    def can(self, obj):
        """
        can only do this if Attached is True
        """

        # For primitive action type, can only carry one object at a time
        if len(self.env.carrying) != 0 and self.env.mode == "primitive":
            assert len(self.env.carrying) == 1
            return False

        # cannot pickup if carrying
        if obj.check_abs_state(self.env, 'inhandofrobot'):
            return False
        
        return super().can(obj) and obj.states['attached'].get_value()

    def do(self, obj):
        super().do(obj)
        objs = self.env.grid.get_all_objs(*obj.cur_pos)
        for toy in objs:
            # Find the toy inside
            if toy.get_name() != obj.get_name():
                self.env.carrying.add(toy) # carry toy
                self.env.grid.remove(*obj.cur_pos, toy) # remove the toy from inside other object
                toy.states['inside'].set_value(obj, False)
                found = True

        obj.states['attached'].set_value(False)

        # check dependencies
        assert found


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
        self.key = 'drop_in'

    def can(self, drop_obj, container_obj, arm):
        """
        can drop obj in if:
        - agent is carrying obj
        - object to be dropped in is not at capacity
        """
        if not super().can(drop_obj):
            return False

        if drop_obj.check_abs_state(self.env, 'in' + arm + 'handofrobot'):
            return False
        
        if container_obj.states['contains'].get_num_objs() > container_obj.max_contain:
            return False

        return True

    def do(self, drop_obj, container_obj, arm):
        # drop
        super().do(drop_obj)
        self.env.carrying[arm].discard(drop_obj)
        container_obj.states['contains'].add_obj(drop_obj)
            

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
        if obj.states['contains'].get_value == 0:
            return False

        if len(self.env.carrying[arm]) != 0:
            return False
        
        return True
    
    def do(self, obj, arm):
        super().do(obj)
        self.env.carrying[arm].add(obj['contains'].remove_obj())



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
        # ** Why not just check if object is in self.carrying[arm]? **
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

        # We need to remove "inside"
        fwd_pos = self.env.front_pos
        furniture = self.env.grid.get_furniture(*fwd_pos, dim)
        if furniture is not None:
            obj.states['inside'].set_value(furniture, False)

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
            assert len(self.env.carryiong[arm]) == 1
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
            assert len(self.env.carryiong[arm]) == 1
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
        dims = self.drop_dims(check_pos)
        return int(0) in dims # Resort to dim = 0 since dims are not really used anymore
    
    def do(self, obj, dim, arm, pos):
        assert self.can(obj, arm, pos)

        self.env.carrying[arm].discard(obj)

        two_fwd_pos = pos + self.env.dir_vec
        throw_pos = pos if 0 in two_fwd_pos or two_fwd_pos[0] == self.env.width - 1 or two_fwd_pos[1] == self.env.height - 1 else two_fwd_pos # boundary condition
        print(throw_pos)
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
        cur = obj.check_abs_state(self.env, 'toggled')
        obj.states['toggled'].set_value(not cur)
        if any(substring in obj.get_name() for substring in ["winnie_cabinet", "piggie_bank"]):
            open_cur = obj.check_abs_state(self.env, 'open')
            obj.states['open'].set_value(not open_cur)
        if any(substring in obj.get_name() for substring in ["farm_toy"]):
            popup_cur = obj.check_abs_state(self.env, 'popup')
            obj.states['popup'].set_value(not popup_cur)
        


