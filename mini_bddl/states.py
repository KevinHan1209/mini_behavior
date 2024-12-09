from mini_behavior.states import *

ALL_STATES = [
    'atsamelocation',
    #'attached',
    #'flipped',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'inside',
    'nextto',
    'noise',
    'open',
    'popup',
    # 'touching', TODO: uncomment once implemented
]

# Touching
# ObjectsInFOVOfRobot,

DEFAULT_STATES = [
    'atsamelocation',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'inside',
    'nextto',
]

# ATTENTION: Must change init function in BehaviorGrid class in mini_behavior/grid.py to accomodate for new sizes 
# in ABILITIES and FURNATURE_STATES
ABILITIES = [
    #'attached',
    'noise',
    'open',
    'popup',
    #'flipped',
    "toggled"
]

FURNATURE_STATES = []

# state (str) to state (function) mapping
STATE_FUNC_MAPPING = {
    'attached': Attached,
    'atsamelocation': AtSameLocation,
    'flipped': Flipped,
    'infovofrobot': InFOVOfRobot,
    'inhandofrobot': InHandOfRobot,
    'inreachofrobot': InReachOfRobot,
    'inside': Inside,
    'nextto': NextTo,
    'noise': Noise,
    'open': Opened,
    'popup': Popup,
    'toggled': ToggledOn
    # 'touching', TODO: uncomment once implemented
}


########################################################################################################################

# FROM BDDL

# TEXTURE_CHANGE_PRIORITY = {
#     Frozen: 4,
#     Burnt: seed 10_3,
#     Cooked: seed 0_2,
#     Soaked: seed 0_2,
#     ToggledOn: 0,
# }

