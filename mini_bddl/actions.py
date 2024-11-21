from mini_behavior.actions import *

ALL_ACTIONS = ['pickup', 'drop', 'drop_in', 'drop_on', 'drop_under', 'toggle', 'shake/bang' ]
DEFAULT_ACTIONS = []

ACTION_FUNC_MAPPING = {
    'pickup': Pickup,
    'drop': Drop,
    'drop_in': DropIn,
    'toggle': Toggle,
    'shake/bang': Shake_Bang
}

CONTROLS = ['left', 'right', 'forward']  # 'down'

