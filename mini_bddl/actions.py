from mini_behavior.actions import *

ALL_ACTIONS = ['pickup', 'drop', 'toggle', 'shake_bang', 'throw']
DEFAULT_ACTIONS = []

ACTION_FUNC_MAPPING = {
    'pickup': Pickup,
    'drop': Drop,
    'toggle': Toggle,
    'shake_bang': Shake_Bang,
    'throw': Throw
}

CONTROLS = ['left', 'right', 'forward', 'kick']  # 'down'

