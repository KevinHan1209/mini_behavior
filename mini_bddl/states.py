from mini_behavior.states import *

ALL_STATES = [
    'atsamelocation',
    'attached',
    'infovofrobot',
    'inlefthandofrobot',
    'inrighthandofrobot',
    'inleftreachofrobot',
    'inrightreachofrobot',
    'inside',
    'kicked',
    'nextto',
    'noise',
    'open',
    'popup',
    'thrown',
    "toggled"
    'pullshed',
    'climbed',
    'contains',
    'hitter',
    'gothit',
    'mouthed',
    'usebrush'
]

# Touching
# ObjectsInFOVOfRobot,

DEFAULT_STATES = [
    'atsamelocation',
    'infovofrobot',
    'inlefthandofrobot',
    'inrighthandofrobot',
    'inleftreachofrobot',
    'inrightreachofrobot',
    'inside',
    'nextto',
]

# ATTENTION: Must change init function in BehaviorGrid class in mini_behavior/grid.py to accomodate for new sizes 
# in ABILITIES and FURNATURE_STATES
ABILITIES = [
    'attached',
    'climbed',
    'contains',
    'kicked',
    'noise',
    'inside',
    'attached',
    'open',
    'popup',
    "thrown",
    "toggled",
    'pullshed',
    'hitter',
    'mouthed',
    'gothit',
    'usebrush'
]

FURNATURE_STATES = []

# state (str) to state (function) mapping
STATE_FUNC_MAPPING = {
    'attached': Attached,
    'atsamelocation': AtSameLocation,
    'climbed': Climbed,
    'contains': Contains,
    'flipped': Flipped,
    'hitter' : Hitter,
    'gothit': GotHit,
    'infovofrobot': InFOVOfRobot,
    'inrighthandofrobot': InRightHandOfRobot,
    'inlefthandofrobot': InLeftHandOfRobot,
    'inleftreachofrobot': InLeftReachOfRobot,
    'inrightreachofrobot': InRightReachOfRobot,
    'inside': Inside,
    'kicked': Kicked,
    'mouthed': Mouthed,
    'nextto': NextTo,
    'noise': Noise,
    'open': Opened,
    'pullshed': Pullshed,
    'popup': Popup,
    'thrown': Thrown,
    'toggled': ToggledOn,
    'usebrush': UseBrush
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

