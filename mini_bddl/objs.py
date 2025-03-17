OBJECTS = [
    # Keys from the dictionary
    "alligator_busy_box", "ball", "beach_ball", "broom", "broom_set", "bucket_toy", "cart_toy", "coin", "cube", "cube_cabinet", 
    "farm_toy", "gear", "gear_toy", "mini_broom", "music_toy", "piggie_bank", "rattle", "red_spiky_ball", 
    "ring_toy", "shape_toy", "shape_sorter", "stroller", "tree_busy_box", "winnie", "winnie_cabinet",
]

FURNITURE = []
# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'alligator_busy_box': 2,
    'ball': 3,
    'beach_ball': 4,
    'broom': 5,
    'broom_set': 6,
    'bucket_toy': 7,
    'cart_toy': 8,
    'coin': 9,
    'cube': 10,
    'cube_cabinet': 11,
    'farm_toy': 12,
    'gear': 13,
    'gear_toy': 14,
    'mini_broom': 15,
    'music_toy': 16,
    'piggie_bank': 17,
    'rattle': 18,
    'red_spiky_ball': 19,
    'ring_toy': 20,
    'shape_toy': 21, 
    'shape_sorter': 22,
    'stroller': 23,
    'tree_busy_box': 24,
    'wall': 25,
    "winnie": 26,
    'winnie_cabinet': 27
}



IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


OBJECT_TO_STR = {
    "alligator_busy_box": "A",
    "ball": "B",
    "beach_ball": "B",
    "broom": "B",
    "broom_set": "B",
    "bucket_toy": "B",
    "cart_toy": "C",
    "coin": "C",
    "cube": "C",
    "cube_cabinet": "C",
    "farm_toy": "F",
    "gear": "G",
    "gear_toy": "G",
    "mini_broom": "M",
    "music_toy": "M",
    "piggie_bank": "P",
    "rattle": "R",
    "red_spiky_ball": "R",
    "ring_toy": "R",
    "shape_toy": "S",
    "shape_sorter": "S",
    "stroller": "S",
    "tree_busy_box": "T",
    "wall": "W",
    "winnie": "W",
    "winnie_cabinet": "W",
}


FURNITURE_CANNOT_ON = []
