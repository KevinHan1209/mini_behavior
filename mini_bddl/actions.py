from mini_behavior.actions import *

ALL_ACTIONS = ['pickup', 'drop', 'toggle', 'shake/bang' ]
DEFAULT_ACTIONS = []

ACTION_FUNC_MAPPING = {
    'pickup': Pickup,
    'drop': Drop,
    'toggle': Toggle,
    'shake_bang': Shake_Bang
}

CONTROLS = ['left', 'right', 'forward']  # 'down'

