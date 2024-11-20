from mini_behavior.states import *

ALL_STATES = [
    'atsamelocation',
    'deform',
    'detach',
    'hidden',
    'infovofrobot',
    'inhandofrobot',
    'inreachofrobot',
    'insameroomasrobot',
    'inside',
    'nextto',
    'noise',
    'onfloor',
    'onTop',
    'popup',
    'reattach',
    'takeout',
    'under',
    ''
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
    'deform',
    'detach',
    'hidden',
    'noise',
    'popup',
    'reattach',
    'takeout',
]

FURNATURE_STATES = []

# state (str) to state (function) mapping
STATE_FUNC_MAPPING = {
    'atsamelocation': AtSameLocation,
    'cleaningTool': CleaningTool,
    'deform': ,
    'detach': ,
    'hidden': ,
    'infovofrobot': InFOVOfRobot,
    'inhandofrobot': InHandOfRobot,
    'inreachofrobot': InReachOfRobot,
    'insameroomasrobot': InSameRoomAsRobot,
    'inside': Inside,
    'nextto': NextTo,
    'noise': ,
    'onfloor': OnFloor,
    'onTop': OnTop,
    'popup': ,
    'reattach': ,
    'takeout': ,
    'under': Under
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

