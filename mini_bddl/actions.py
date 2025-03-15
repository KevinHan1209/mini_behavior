from mini_behavior.actions import *

ALL_ACTIONS = ['pickup', 'drop', 'toggle', 'noise_toggle', 'throw', 'push', 'pull', 'takeout', 'dropin', 'assemble', 'disassemble', 'hit', 'hitwithobject']
DEFAULT_ACTIONS = []

ACTION_FUNC_MAPPING = {
    'pickup': Pickup,
    'drop': Drop,
    'toggle': Toggle,
    'noise_toggle': NoiseToggle,
    'throw': Throw,
    'push': Push,
    'pull': Pull,
    'takeout': TakeOut,
    'dropin': DropIn,
    'assemble': Assemble,
    'disassemble': Disassemble,
    'hit' : Hit,
    'hitwithobject' : HitWithObject
}

CONTROLS = ['left', 'right', 'forward', 'kick']  # 'down'

