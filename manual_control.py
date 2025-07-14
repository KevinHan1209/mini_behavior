#!/usr/bin/env python3

import argparse
from gym_minigrid.wrappers import *
from mini_behavior.window import Window
from mini_behavior.utils.save import get_step, save_demo
from mini_behavior.grid import GridDimension
import numpy as np
import mini_behavior.envs.picking_up_a_rattle
import mini_behavior.envs.multitoy
from mini_behavior.register import register

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False
ROOM_SIZE = 8
MAX_STEPS = 1000
env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
register(
    id='MiniGrid-MultiToy-8x8-N2-v0',
    entry_point='mini_behavior.envs:MultiToyEnv',
    kwargs=env_kwargs
)


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)


def render_furniture():
    global show_furniture
    show_furniture = not show_furniture

    if show_furniture:
        img = np.copy(env.furniture_view)

        # i, j = env.agent.cur_pos
        i, j = env.agent_pos
        ymin = j * TILE_PIXELS
        ymax = (j + 1) * TILE_PIXELS
        xmin = i * TILE_PIXELS
        xmax = (i + 1) * TILE_PIXELS

        img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
            img[ymin:ymax, xmin:xmax, :], env.agent_dir)
        img = env.render_furniture_states(img)

        window.show_img(img)
    else:
        obs = env.gen_obs()
        redraw(obs)


def show_states():
    imgs = env.render_states()
    window.show_closeup(imgs)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def load():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    prev_obs = env.gen_obs()
    obs, reward, done, info = env.step(action)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if args.save:
        all_steps[env.step_count] = (prev_obs, action)

    if done:
        print('done!')
        if args.save:
            save_demo(all_steps, args.env, env.episode)
        reset()
    else:
        redraw(obs)


def switch_dim(dim):
    env.switch_dim(dim)
    print(f'switching to dim: {env.render_dim}')
    obs = env.gen_obs()
    redraw(obs)


def key_handler_cartesian(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    if event.key == 'o':
        step(env.actions.kick)
        return
    if event.key == 'c':
        step(env.actions.climb)
        return
    # Spacebar
    if event.key == ' ':
        render_furniture()
        return
    if event.key == 'pageup':
        step('choose')
        return
    if event.key == 'enter':
        env.save_state()
        return
    if event.key == 'pagedown':
        show_states()
        return
    if event.key == '0':
        switch_dim(None)
        return
    if event.key == '1':
        switch_dim(0)
        return
    if event.key == '2':
        switch_dim(1)
        return
    if event.key == '3':
        switch_dim(2)
        return
    
action_list = []
def key_handler_primitive(event):
    global action_list
    # Escape action to exit
    if event.key == 'escape':
        window.close()
        return
    # Pagina down action to show states
    elif event.key == 'pagedown':
        show_states()
        return
    if len(action_list) < 2:
        print("Select object manipulation action")

        valid_action = False  # Flag to check if the action is valid
        
        # Object manipulation actions
        if event.key == '0':
            action_list.append(env.manipulation_actions.pickup_0)
            valid_action = True
        elif event.key == '1':
            action_list.append(env.manipulation_actions.drop_0)
            valid_action = True
        elif event.key == 't':
            action_list.append(env.manipulation_actions.toggle)
            valid_action = True
        elif event.key == 'p':
            action_list.append(env.manipulation_actions.throw_0)
            valid_action = True
        elif event.key == 'w':
            action_list.append(env.manipulation_actions.push)
            valid_action = True
        elif event.key == 'e':
            action_list.append(env.manipulation_actions.pull)
            valid_action = True
        elif event.key == 'z':
            action_list.append(env.manipulation_actions.takeout)
            valid_action = True
        elif event.key == 'x':
            action_list.append(env.manipulation_actions.dropin)
            valid_action = True
        elif event.key == 'm':
            action_list.append(env.manipulation_actions.mouthing)
            valid_action = True
        elif event.key == 'b':
            action_list.append(env.manipulation_actions.brush)
            valid_action = True
        elif event.key == '3':
            action_list.append(env.manipulation_actions.assemble)
            valid_action = True
        elif event.key == '4':
            action_list.append(env.manipulation_actions.disassemble)
            valid_action = True
        elif event.key == '5':
            action_list.append(env.manipulation_actions.hit)
            valid_action = True
        elif event.key == '6':  
            action_list.append(env.manipulation_actions.hitwithobject)
            valid_action = True
        if not valid_action:
            print("Invalid object manipulation action, please try again.")

    elif len(action_list) == 2:
        print("Select locomotive action (left/right/up/kick)")
        valid_action = False  # Reset flag for locomotive actions
        
        # Locomotion actions
        if event.key == 'left':
            action_list.append(env.locomotion_actions.left)
            valid_action = True
        elif event.key == 'right':
            action_list.append(env.locomotion_actions.right)
            valid_action = True
        elif event.key == 'up':
            action_list.append(env.locomotion_actions.forward)
            valid_action = True
        elif event.key == '2':
            action_list.append(env.locomotion_actions.kick)
            valid_action = True
        elif event.key == 'c':
            action_list.append(env.locomotion_actions.climb)
            valid_action = True

        if not valid_action:
            print("Invalid locomotive action, please select from left, right, up, or kick.")
            
    elif len(action_list) == 3:
        print("Final action: ", action_list)
        step(action_list)
        action_list = []
    print("Selected", event.key)
    return



parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiToy-8x8-N2-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
# NEW
parser.add_argument(
    "--save",
    default=False,
    help="whether or not to save the demo_16"
)
# NEW
parser.add_argument(
    "--load",
    default=None,
    help="path to load state from"
)

args = parser.parse_args()

env = gym.make(args.env)
env.teleop_mode()
if args.save:
    # We do not support save for cartesian action space
    assert env.mode == "primitive"

all_steps = {}

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('mini_behavior - ' + args.env)
if env.mode == "cartesian":
    window.reg_key_handler(key_handler_cartesian)
elif env.mode == "primitive":
    window.reg_key_handler(key_handler_primitive)

if args.load is None:
    reset()
else:
    load()

# Blocking event loop
window.show(block=True)
