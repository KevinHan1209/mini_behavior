from mini_behavior.states import *

ALL_STATES = [
    'atsamelocation',
    #'attached',
    #'flipped',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'insameroomasrobot',
    'inside',
    'nextto',
    'noise',
    'onfloor',
    'onTop',
    'open',
    'popup',
    'under',
    # 'touching', TODO: uncomment once implemented
]

# Touching
# ObjectsInFOVOfRobot,

DEFAULT_STATES = [
    'atsamelocation',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'insameroomasrobot',
    'inside',
    'nextto',
    'onfloor',
    'onTop',
    'under'
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
    'insameroomasrobot': InSameRoomAsRobot,
    'inside': Inside,
    'nextto': NextTo,
    'noise': Noise,
    'onfloor': OnFloor,
    'onTop': OnTop,
    'open': Opened,
    'popup': Popup,
    'under': Under,
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

