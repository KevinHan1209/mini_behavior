import sys

import gymnasium as gym
import minigrid

# Expose gymnasium under the legacy "gym" name for callers expecting gym.
sys.modules.setdefault("gym", gym)

# Provide minimal compatibility aliases for legacy gym_minigrid imports.
sys.modules.setdefault("gym_minigrid", minigrid)
sys.modules.setdefault("gym_minigrid.wrappers", minigrid.wrappers)
sys.modules.setdefault("gym_minigrid.envs", minigrid.envs)
import minigrid.utils.rendering as _mg_render
sys.modules.setdefault("gym_minigrid.rendering", _mg_render)

# Import envs lazily when needed to avoid side effects at import time.
