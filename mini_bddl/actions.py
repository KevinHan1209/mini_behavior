from mini_behavior.actions import *

ALL_ACTIONS = ['pickup', 'drop', 'toggle', 'noise_toggle', 'throw', 'push', 'pull', 'takeout', 'dropin', 'assemble', 'disassemble', 'hit', 'hitwithobject', 'brush']
DEFAULT_ACTIONS = []

ACTION_FUNC_MAPPING = {
    'brush': Brush,
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
    'hitwithobject' : HitWithObject,
    'mouthing': Mouthing
}

CONTROLS = ['left', 'right', 'forward', 'kick', 'climb']  # 'down'

