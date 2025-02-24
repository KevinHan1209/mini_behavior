# MODIFIED FROM MINIGRID REPO

import os
import pickle as pkl
from enum import IntEnum
from gym import spaces
from gym_minigrid.minigrid import MiniGridEnv
from mini_bddl.actions import ACTION_FUNC_MAPPING
from .objects import *
from .grid import BehaviorGrid, GridDimension, is_obj
from mini_behavior.window import Window
from .utils.utils import AttrDict, get_total_states
from mini_behavior.states import RelativeObjectState
from mini_behavior.actions import Pickup, Drop, Toggle, Open, Close

import numpy as np
import torch

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32


class MiniBehaviorEnv(MiniGridEnv):
    """
    2D grid world game environment
    """
    metadata = {
        # Deprecated: use 'render_modes' instead
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,  # Deprecated: use 'render_fps' instead
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    '''
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        toggle = 3
        open = 4
        close = 5
        slice = 6
        cook = 7
        drop_in = 8
        pickup_0 = 9
        pickup_1 = 10
        pickup_2 = 11
        drop_0 = 12
        drop_1 = 13
        drop_2 = 14
    '''
    class ObjectActions(IntEnum):
        toggle = 0
        shake_bang = 1
        pickup_0 = 2
        drop_0 = 3

    class LocoActions(IntEnum):
        left = 0
        right = 1
        forward = 2
        kick = 3


    def __init__(
        self,
        test_env,
        mode,
        width,
        height,
        num_objs,
        max_steps,
        grid_size=None,
        see_through_walls=True,
        seed=1337,
        agent_view_size=7,
        highlight=True,
        tile_size=TILE_PIXELS,
        dense_reward=False,
        kick_length = 1
    ):
        self.test_env = test_env
        if self.test_env:
            self.exploration_logger = []
        self.episode = 0
        self.teleop = False  # True only when set manually
        self.last_action = None
        self.action_done = None

        self.render_dim = None

        self.highlight = highlight
        self.tile_size = tile_size
        self.dense_reward = dense_reward
        self.kick_length = kick_length

        # Initialize the RNG
        self.seed(seed=seed)
        self.furniture_view = None

        if num_objs is None:
            num_objs = {}

        self.objs = {}
        self.obj_instances = {}

        # Right now this can only be changed through the wrapper
        self.use_full_obs = False

        locomotion_actions = ["left", "right", "forward", "kick"]
        manipulation_actions = []  # Actions for both arms

        for obj_type in num_objs.keys():
            self.objs[obj_type] = []
            for i in range(num_objs[obj_type]):
                obj_name = '{}_{}'.format(obj_type, i)

                if obj_type in OBJECT_CLASS.keys():
                    obj_instance = OBJECT_CLASS[obj_type](name=obj_name)
                else:
                    obj_instance = WorldObj(obj_type, None, obj_name)

                self.objs[obj_type].append(obj_instance)
                self.obj_instances[obj_name] = obj_instance

            # Get manipulation actions for this object type
            applicable_actions = obj_instance.actions
            for action_name in applicable_actions:
                manipulation_actions.append(obj_type + "/" + action_name)

        # Store separate action lists
        self.locomotion_actions = MiniBehaviorEnv.LocoActions
        self.manipulation_actions = MiniBehaviorEnv.ObjectActions

        # Use MultiDiscrete action space (3D)
        self.action_space = spaces.MultiDiscrete([
            len(manipulation_actions),  # First dimension: left arm actions
            len(manipulation_actions),  # Second dimension: right arm actions
            len(locomotion_actions)     # Third dimension: locomotion actions
        ])


        super().__init__(grid_size=grid_size,
                         width=width,
                         height=height,
                         max_steps=max_steps,
                         see_through_walls=see_through_walls,
                         agent_view_size=agent_view_size,
                         )

        self.grid = BehaviorGrid(width, height)

        pixel_dim = self.grid.pixel_dim

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, pixel_dim),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        self.mode = mode

        self.carrying = {key: set() for key in ['left', 'right']}



    def copy_objs(self):
        from copy import deepcopy
        return deepcopy(self.objs), deepcopy(self.obj_instances)

    # TODO: check this works
    def load_objs(self, state):
        obj_instances = state['obj_instances']
        grid = state['grid']
        for obj in self.obj_instances.values():
            if type(obj) != Wall and type(obj) != Door:
                load_obj = obj_instances[obj.name]
                obj.load(load_obj, grid, self)

        for obj in self.obj_instances.values():
            obj.contains = []
            for other_obj in self.obj_instances.values():
                if other_obj.check_rel_state(self, obj, 'inside'):
                    obj.contains.append(other_obj)

    # TODO: check this works
    def get_state(self):
        grid = self.grid.copy()
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        objs, obj_instances = self.copy_objs()
        state = {'grid': grid,
                 'agent_pos': agent_pos,
                 'agent_dir': agent_dir,
                 'objs': objs,
                 'obj_instances': obj_instances
                 }
        return state

    def save_state(self, out_file='cur_state.pkl'):
        state = self.get_state()
        with open(out_file, 'wb') as f:
            pkl.dump(state, f)
            print(f'saved to: {out_file}')

    # TODO: check this works
    def load_state(self, load_file):
        assert os.path.isfile(load_file)
        with open(load_file, 'rb') as f:
            state = pkl.load(f)
            self.load_objs(state)
            self.grid.load(state['grid'], self)
            self.agent_pos = state['agent_pos']
            self.agent_dir = state['agent_dir']
        return self.grid

    def reset(self):
        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        self.carrying = {key: set() for key in ['left', 'right']}

        for obj in self.obj_instances.values():
            obj.reset()

        self.reward = 0

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        self.update_states()
        # generate furniture view
        self.furniture_view = self.grid.render_furniture(tile_size=TILE_PIXELS, obj_instances=self.obj_instances)
        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        assert self.grid.is_empty(*self.agent_pos)

        # Step count since episode start
        self.step_count = 0
        self.episode += 1

        # For keeping track of dense reward
        self.previous_progress = self.get_progress()

        # Return first observation
        if self.use_full_obs:
            obs = self.gen_full_obs()
        else:
            obs = self.gen_obs()

        if self.test_env:
            self.update_exploration_metrics()

        return obs

    def gen_full_obs(self):
        full_grid = self.grid.encode()
        # Set the agent state and direction as part of the observation
        full_grid[self.agent_pos[0]][self.agent_pos[1]][0] = OBJECT_TO_IDX['agent']
        full_grid[self.agent_pos[0]][self.agent_pos[1]][1] = self.agent_dir

        return {
            'mission': self.mission,
            'image': full_grid
        }

    # Each env can implement its own get_progress function
    def get_progress(self):
        return 0

    def _gen_grid(self, width, height):
        self._gen_objs()
        assert self._init_conditions(), "Does not satisfy initial conditions"
        self.place_agent()

    def _gen_objs(self):
        assert False, "_gen_objs needs to be implemented by each environment"

    def _init_conditions(self):
        print('no init conditions')
        return True

    def _end_conditions(self):
        print('no end conditions')
        return False

    def place_obj_pos(self,
                      obj,
                      pos,
                      top=None,
                      size=None,
                      reject_fn=None
                      ):
        """
        Place an object at a specific position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param obj: the object to place
        :param pos: the top left of the pos we want the object to be placed
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        width = 1 if obj is None else obj.width
        height = 1 if obj is None else obj.height

        valid = True

        if pos[0] < top[0] or pos[0] > min(top[0] + size[0], self.grid.width - width + 1)\
                or pos[1] < top[1] or pos[1] > min(top[1] + size[1], self.grid.height - height + 1):
            raise NotImplementedError(f'position {pos} not in grid')

        for dx in range(width):
            for dy in range(height):
                x = pos[0] + dx
                y = pos[1] + dy

                # Don't place the object on top of another object
                if not self.grid.is_empty(x, y):
                    valid = False
                    break

                # Don't place the object where the agent is
                if np.array_equal((x, y), self.agent_pos):
                    valid = False
                    break

                # Check if there is a filtering criterion
                if reject_fn and reject_fn(self, (x, y)):
                    valid = False
                    break

        if not valid:
            raise ValidationErr(f'failed in place_obj at {pos}')

        self.grid.set(*pos, obj)

        if obj:
            self.put_obj(obj, *pos)

        return pos

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            width = 1 if obj is None else obj.width
            height = 1 if obj is None else obj.height

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width - width + 1)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height - height + 1))
            ))

            valid = True

            for dx in range(width):
                for dy in range(height):
                    x = pos[0] + dx
                    y = pos[1] + dy

                    # If place door, check if it is blocked by other objects
                    if obj is not None and obj.name == "door":
                        left, right, up, down = 0, 0, 0, 0
                        if (x < self.grid.width-1 and not self.grid.is_empty(x+1, y)) or x == self.grid.width - 1:
                            right = 1
                        if (x > 1 and not self.grid.is_empty(x-1, y)) or x == 0:
                            left = 1
                        if (y < self.grid.height-1 and not self.grid.is_empty(x, y+1)) or y == self.grid.height - 1:
                            down = 1
                        if (y > 1 and not self.grid.is_empty(x, y-1)) or y == 0:
                            up = 1
                        if obj.dir=='horz' and (up or down):
                            valid = False
                            break
                        if obj.dir=='vert' and (left or right):
                            valid = False
                            break
                    # Don't place the object on top of another object
                    else:
                        if not self.grid.is_empty(x, y):
                            valid = False
                            break
                    
                    # Don't place the object next to door

                    if obj is not None and obj.type != "door":
                        if x < self.grid.width-1:
                            fur = self.grid.get_furniture(x+1, y)
                            # print("right", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                break
                        if x > 1:
                            fur = self.grid.get_furniture(x-1, y)
                            # print("left", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                break
                        if y < self.grid.height-1:
                            fur = self.grid.get_furniture(x, y+1)
                            # print("down", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                break
                        if y > 1:
                            fur = self.grid.get_furniture(x, y-1)
                            # print("top", fur)
                            if fur is not None and fur.type == "door":
                                valid = False
                                break

                    # Don't place the object where the agent is
                    if np.array_equal((x, y), self.agent_pos):
                        valid = False
                        break

                    # Check if there is a filtering criterion
                    if reject_fn and reject_fn(self, (x, y)):
                        valid = False
                        break

            if not valid:
                continue

            break

        self.grid.set(*pos, obj)

        if obj:
            self.put_obj(obj, *pos)

        return pos

    def put_obj(self, obj, i, j, dim=0):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set(i, j, obj, dim)
        obj.init_pos = (i, j)
        obj.update_pos((i, j))

        if obj.is_furniture():
            for pos in obj.all_pos:
                self.grid.set(*pos, obj, dim)

    def teleop_mode(self):
        self.teleop = True

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = BehaviorGrid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        if obs_grid.is_empty(vx, vy):
            return False

        for i in range(3):
            if [obj.type for obj in obs_cell[i]] != [obj.type for obj in world_cell[i]]:
                return False

        return True

    def step(self, action):
        # Keep track of last action
        self.last_action = action
        self.step_count += 1
        self.action_done = True

        # Parse action components
        manip_action_left, manip_action_right, locomotion_action = action

        # Get the position and contents around the agent
        fwd_pos, right_pos, left_pos, upper_right_pos, upper_left_pos = self.front_pos, self.right_pos, self.left_pos, self.upper_right_pos, self.upper_left_pos
        fwd_cell, right_cell, left_cell, upper_right_cell, upper_left_cell = self.grid.get(*fwd_pos), self.grid.get(*right_pos), self.grid.get(*left_pos), self.grid.get(*upper_right_pos), self.grid.get(*upper_left_pos)
        
        # Establish action sequence
        left_seq = [fwd_cell, upper_left_cell, left_cell]
        right_seq = [fwd_cell, upper_right_cell, right_cell]

        # Separate logic to check if object was dropped in front. For actions with drop and forward in the same action
        was_dropped = False

        ## **Handle Manipulation Actions for Both Arms**
        for arm_action, arm in zip([manip_action_left, manip_action_right], ["left", "right"]):
            seq = left_seq if arm == 'left' else right_seq
            print("CURRENTLY CARRYING: ",self.carrying )
            if arm_action != -1:  # -1 means no action taken
                curr_action = self.manipulation_actions(arm_action)
                action_name = curr_action.name  # Convert index to action name
                if "toggle" not in action_name or "shake/bang" not in action_name:
                    self.silence()
                self.action_done = False

                if "pickup" in action_name or "drop" in action_name:
                    action_dim = action_name.split('_')  # list: [action, dim]
                    if action_name == "drop_in":
                        action_class = ACTION_FUNC_MAPPING["drop_in"]
                    else:
                        action_class = ACTION_FUNC_MAPPING[action_dim[0]]
                else:
                    action_class = ACTION_FUNC_MAPPING[action_name]
                self.action_done = False

                # Pickup involves certain dimension
                if "pickup" in action_name:
                    for cell in seq:
                        #print("Checking Cell:", cell)
                        for obj in cell[int(action_dim[1])]:
                            if is_obj(obj) and action_class(self).can(obj, arm):
                                action_class(self).do(obj, arm)
                                self.action_done = True
                                break
                        if self.action_done:
                            break
                # Drop act on carried object
                elif "drop" in action_name:
                    pos_seq = [fwd_pos, upper_left_pos, left_pos] if arm == 'left' else [fwd_pos, upper_right_pos, right_pos]
                    for obj in self.carrying[arm]:
                        for pos in pos_seq:
                            if action_class(self).can(obj, arm, pos):
                                drop_dim = obj.available_dims
                                print('\nDrop Dim:', drop_dim)
                                if action_dim[1] == "in":
                                    # For drop_in, we don't care about dimension
                                    action_class(self).do(obj, np.random.choice(drop_dim), arm)
                                    self.action_done = True
                                elif int(action_dim[1]) in drop_dim:
                                    action_class(self).do(obj, int(action_dim[1]), arm, pos)
                                    self.action_done = True
                                    was_dropped = True
                                    break
                        if self.action_done:
                            break
                # Everything else act on the forward cell
                else:
                    for cell in seq:
                        for dim in cell:
                            for obj in dim:
                                if is_obj(obj) and action_class(self).can(obj, arm):
                                    action_class(self).do(obj, arm)
                                    self.action_done = True
                                    # Take care of noise state if action is toggle but not on a noisy object
                                    if action_name == "toggle" and not any(substring in obj.get_name() for substring in ["music_toy", "piggie_bank"]):
                                        self.silence()
                                    break
                            if self.action_done:
                                break
                        if self.action_done:
                            self.update_states()
                            break
        ## **Handle Locomotion Actions**
        if locomotion_action == self.locomotion_actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        elif locomotion_action == self.locomotion_actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4  # Rotate right
        elif locomotion_action == self.locomotion_actions.forward:
            can_overlap = True
            for dim in fwd_cell:
                for obj in dim:
                    if is_obj(obj) and not obj.can_overlap:
                        can_overlap = False
                        break
            if can_overlap and not was_dropped:
                self.agent_pos = fwd_pos
            else:
                self.action_done = False
        elif locomotion_action == self.locomotion_actions.kick:
            dim = int(0)
            for obj in fwd_cell[dim]:
                if is_obj(obj) and obj.possible_action('kick'):
                    new_pos = fwd_pos + self.dir_vec * self.kick_length
                    print("Fwd pos:", fwd_pos)
                    print("New pos:", new_pos)
                    dims = self.drop_dims(new_pos)
                    print("dims:", dims)
                    if dims != [] and dim in dims:
                        # modified code from pickup
                        self.grid.remove(*obj.cur_pos, obj)
                        self.grid.set_all_objs(*obj.cur_pos, [None, None, None])
                        obj.update_pos(new_pos)
                        # modified code from drop
                        self.grid.set(*new_pos, obj, dim)
                        break


        self.update_states()
        reward = self._reward()
        done = self._end_conditions() or self.step_count >= self.max_steps
        if self.use_full_obs:
            obs = self.gen_full_obs()
        else:
            obs = self.gen_obs()

        if self.test_env:
            self.update_exploration_metrics()

        return obs, reward, done, {}
    
    def drop_dims(self, pos):
        dims = []

        all_items = self.grid.get_all_items(*pos)
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
    
    def update_exploration_metrics(self):
        objs_dict = {}
        for obj_type in self.objs.values():
            for obj in obj_type:
                states_dict = {}
                for state_value in obj.states:
                    if not isinstance(obj.states[state_value], RelativeObjectState):
                        state = obj.states[state_value].get_value(self)
                        states_dict[state_value] = state
                objs_dict[obj.name] = states_dict
        self.exploration_logger.append(objs_dict)

    def silence(self):
        '''
        Set all noise states to false
        '''
        for obj_type in self.objs.values():
                        for obj in obj_type:
                            for state_name in obj.states:
                                if state_name == "noise":
                                    obj.states[state_name].set_value(False)


    def _reward(self):
        if self._end_conditions():
            return 1
        else:
            # Dense rewards are implemented only for washing_pots_and_pans and putting_away_dishes
            if self.dense_reward:
                cur_progress = self.get_progress()
                reward = cur_progress - self.previous_progress
                self.previous_progress = cur_progress
                return reward
            else:
                return 0

    def all_reachable(self):
        return [obj for obj in self.obj_instances.values() if obj.check_abs_state(self, 'inreachofrobot')]

    def update_states(self):
        for obj in self.obj_instances.values():
            for name, state in obj.states.items():
                if state.type == 'absolute':
                    state._update(self)
        self.grid.state_values = {obj: obj.get_ability_values(self) for obj in self.obj_instances.values()}

    def render(self, mode='human', highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        if mode == "human" and not self.window:
            self.window = Window("mini_behavior")
            self.window.show(block=False)

        img = super().render(mode='rgb_array', highlight=highlight, tile_size=tile_size)

        if self.render_dim is None:
            img = self.render_furniture_states(img)
        else:
            img = self.render_furniture_states(img, dim=self.render_dim)

        if self.window:
            self.window.set_inventory(self)

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def render_states(self, tile_size=TILE_PIXELS):
        pos = self.front_pos
        imgs = []
        furniture = self.grid.get_furniture(*pos)
        img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
        if furniture:
            furniture.render(img)
            state_values = furniture.get_ability_values(self)
            GridDimension.render_furniture_states(img, state_values)
        imgs.append(img)

        for grid in self.grid.grid:
            furniture, obj = grid.get(*pos)
            state_values = obj.get_ability_values(self) if obj else None
            img = GridDimension.render_tile(furniture, obj, state_values, draw_grid_lines=False)
            imgs.append(img)

        return imgs

    def render_furniture_states(self, img, tile_size=TILE_PIXELS, dim=None):
        for obj in self.obj_instances.values():
            if obj.is_furniture():
                if dim is None or dim in obj.dims:
                    i, j = obj.cur_pos
                    ymin = j * tile_size
                    ymax = (j + obj.height) * tile_size
                    xmin = i * tile_size
                    xmax = (i + obj.width) * tile_size
                    sub_img = img[ymin:ymax, xmin:xmax, :]
                    state_values = obj.get_ability_values(self)
                    GridDimension.render_furniture_states(sub_img, state_values)
        return img

    def switch_dim(self, dim):
        self.render_dim = dim
        self.grid.render_dim = dim

    def get_total_exploration_metric(self):
        '''
        Returns all information about state exploration
        Format: [{obj.name: {state : value}}]
        '''
        if self.test_env:
            return self.exploration_logger
        else:
            return None
        

    def get_exploration_statistics(self):
        '''
        Returns percentage of total state space explored per object.
        Aggregates state-value pairs across all timesteps.
        Format: {obj.name: % of state space explored}
        '''
        # Dictionary to store aggregated state-value pairs per object
        aggregated_states = {}
        print(self.exploration_logger)
        # Iterate through each timestep's dictionary in the exploration logger
        for timestep_dict in self.exploration_logger:
            for obj, states in timestep_dict.items():
                if obj not in aggregated_states:
                    aggregated_states[obj] = set()  # Use a set to track unique state-value pairs
                for state, value in states.items():
                    aggregated_states[obj].add((state, value))  # Add state-value pair to the set

        # Calculate the exploration percentage per object
        percentage_explored = {}
        for obj_object, (obj, unique_pairs) in zip(list(obj for obj_type in self.objs.values() for obj in obj_type), aggregated_states.items()):
            total_states = get_total_states(obj_object)
            explored = len(unique_pairs)
            percentage_explored[obj] = "{:.2f}".format((explored / total_states) * 100)

        return percentage_explored
    

    def get_exploration_statistics2(self):
        '''
        Returns percentage of total state space explored per object based on CHANGES IN STATES ONLY.
        In other words, doesn't count default states towards total state space
        Format: {obj.name: % of state space explored}
        '''
        # Initialize tracking of explored states (both True and False)
        default_states = self.exploration_logger[0]
        visited_states = {obj: {state: set() for state in states} for obj, states in default_states.items()}

        # Iterate through timesteps to track state values
        for timestep_dict in self.exploration_logger:
            for obj, states in timestep_dict.items():
                for state, value in states.items():
                    # Record the value (True or False) for this state
                    visited_states[obj][state].add(value)

        # Calculate the percentage explored per object
        percentage_explored = {}
        for obj, states in visited_states.items():
            total_states = len(states)
            explored_count = sum(1 for state, values in states.items() if len(values) == 2)  # Both True and False visited
            percentage_explored[obj] = (explored_count / total_states) * 100

        return percentage_explored
    
    def get_state_exploration_counts(self):
        '''
        Returns a dictionary which returns the number of True/False occurrences for each state in each object
        Format: {object: {state: {True/False: # of occurrences}}}
        '''
        from collections import defaultdict

        # Initialize a nested dictionary to store the counts for each state per object
        state_counts = defaultdict(lambda: defaultdict(lambda: {"True": 0, "False": 0}))

        # Iterate through each timestep in the exploration logger
        for timestep_dict in self.exploration_logger:
            for obj_name, states in timestep_dict.items():
                for state_name, state_value in states.items():
                    # Increment the count based on the state value
                    state_counts[obj_name][state_name][str(state_value)] += 1

        # Convert defaultdict to a standard dictionary for better readability (optional)
        state_counts = {obj: dict(states) for obj, states in state_counts.items()}

        return state_counts

