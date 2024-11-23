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

    def can(self, obj):
        """
        check if possible to do action
        """

        # check if possible to do the action on the object
        if not obj.possible_action(self.key):
            return False

        # check if the object is in reach of the agent
        if not obj.check_abs_state(self.env, 'inreachofrobot'):
            return False

        return True

    def do(self, obj):
        """
        do action
        """
        assert self.can(obj), 'Cannot perform action'

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
            if obj.get_name() == "broom_set" and find_tool(self.env, ["broom"]):
                return True
        return False

    def do(self, obj):
        super().do(obj)
        obj.states['attached'].set_value(True)
        # find correct tool that is in hand of the agent
        if obj.get_name() == "gear_toy":
            toys = self.env.objs.get("gear", []) 
        elif obj.get_name() == "broom_set":
            toys = self.env.objs.get("broom, []")
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


class Cook(BaseAction):
    def __init__(self, env):
        super(Cook, self).__init__(env)
        self.key = 'cook'
        self.tools = ['pan']
        self.heat_sources = ['stove']

    def can(self, obj):
        """
        can perform action if:
        - obj is cookable
        - agent is carrying a cooking tool
        - agent is infront of a heat source
        - the heat source is toggled on
        """
        if not super().can(obj):
            return False

        if find_tool(self.env, self.tools):
            front_cell = self.env.grid.get_all_items(*self.env.agent_pos)
            for obj2 in front_cell:
                if obj2 is not None and obj2.type in self.heat_sources:
                    return obj2.check_abs_state(self.env, 'toggleable')
        return False

    def do(self, obj):
        super().do(obj)
        obj.states['cookable'].set_value(True)


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

    def can(self, obj):
        """
        can drop obj if:
        - agent is carrying obj
        - there is no obj in base of forward cell
        """
        if not super().can(obj):
            return False

        if not obj.check_abs_state(self.env, 'inhandofrobot'):
            return False

        fwd_pos = self.env.front_pos
        dims = self.drop_dims(fwd_pos)
        obj.available_dims = dims

        return dims != []

    def do(self, obj, dim):
        super().do(obj)

        self.env.carrying.discard(obj)

        fwd_pos = self.env.front_pos

        # change object properties
        obj.cur_pos = fwd_pos
        # change agent / grid
        self.env.grid.set(*fwd_pos, obj, dim)


class DropIn(BaseAction):
    def __init__(self, env):
        super(DropIn, self).__init__(env)
        self.key = 'drop_in'

    def drop_dims(self, pos):
        dims = []

        all_items = self.env.grid.get_all_items(*pos)
        last_furniture, last_obj = 'floor', 'floor'
        for i in range(3):
            furniture = all_items[2*i]
            obj = all_items[2*i + 1]

            if obj is None and furniture is not None and furniture.can_contain and i in furniture.can_contain:
                if 'openable' not in furniture.states or furniture.check_abs_state(self.env, 'openable'):
                    if last_obj is not None:
                        dims.append(i)

            last_furniture = furniture
            last_obj = obj
        return dims

    def can(self, obj):
        """
        can drop obj under if:
        - agent is carrying obj
        - middle of forward cell is open
        - obj does not contain another obj
        """
        if not super().can(obj):
            return False

        if not obj.check_abs_state(self.env, 'inhandofrobot'):
            return False

        fwd_pos = self.env.front_pos
        dims = self.drop_dims(fwd_pos)
        obj.available_dims = dims

        return dims != []

    def do(self, obj, dim):
        # drop
        super().do(obj)
        self.env.carrying.discard(obj)

        fwd_pos = self.env.front_pos
        obj.cur_pos = fwd_pos
        self.env.grid.set(*fwd_pos, obj, dim)

        # drop in and update
        furniture = self.env.grid.get_furniture(*fwd_pos, dim)
        obj.states['inside'].set_value(furniture, True)


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

    def can(self, obj):
        if not super().can(obj):
            return False

        # For primitive action type, can only carry one object at a time
        if len(self.env.carrying) != 0 and self.env.mode == "primitive":
            assert len(self.env.carrying) == 1
            return False

        # cannot pickup if carrying
        if obj.check_abs_state(self.env, 'inhandofrobot'):
            return False

        return True

    def do(self, obj):
        super().do(obj)
        self.env.carrying.add(obj)

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
        assert obj.check_abs_state(self.env, 'inhandofrobot')
        assert not obj.check_abs_state(self.env, 'onfloor')


class Slice(BaseAction):
    def __init__(self, env):
        super(Slice, self).__init__(env)
        self.key = 'slice'
        self.slicers = ['carving_knife', 'knife']

    def can(self, obj):
        """
        can perform action if:
        - action is sliceable
        - agent is holding a slicer
        """
        if not super().can(obj):
            return False
        return find_tool(self.env, self.slicers)

    def do(self, obj):
        super().do(obj)
        obj.states['sliceable'].set_value()

class Shake_Bang(BaseAction):
    def __init__(self, env):
        super(Shake_Bang, self).__init__(env)
        self.key =  'shake_bang'
        
    def do(self, obj):
        super().do(obj)
        obj.states['noise'].set_value(True)
        


class Toggle(BaseAction):
    def __init__(self, env):
        super(Toggle, self).__init__(env)
        self.key = 'toggle'

    def do(self, obj):
        """
        toggle from on to off, or off to on
        """
        super().do(obj)
        cur = obj.check_abs_state(self.env, 'toggled')
        obj.states['toggled'].set_value(not cur)
        if obj.get_name() in ["music_toy", "piggie_bank"] :
            obj.states['noise'].set_value(True)
        if obj.get_name() in ["winnie_cabinet", "piggie_bank"]:
            open_cur = obj.check_abs_state(self.env, 'open')
            obj.states['open'].set_value(not open_cur)
        if obj.get_name() in ["farm_toy"]:
            popup_cur = obj.check_abs_state(self.env, 'popup')
            obj.states['popup'].set_value(not popup_cur)
        


