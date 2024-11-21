OBJECTS = [
    # Keys from the dictionary
    "alligator_busy_box", "ball", "beach_ball", "broom", "broom_set", "bucket_toy", "cart_toy", "coin", "cube", "cube_cabinet", 
    "farm_toy", "gear", "gear_pole", "music_toy", "piggie_bank", "rattle", "red_spiky_ball", 
    "ring_toy", "shape_sorter", "stroller", "tree_busy_box", "winnie", "winnie_cabinet",
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
    'gear_pole': 14,
    'music_toy': 15,
    'piggie_bank': 16,
    'rattle': 17,
    'red_spiky_ball': 18,
    'ring_toy': 19,
    'shape_sorter': 20,
    'stroller': 21,
    'tree_busy_box': 22,
    "winnie": 23,
    'winnie_cabinet': 24
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
    "gear_pole": "G",
    "music_toy": "M",
    "piggie_bank": "P",
    "rattle": "R",
    "red_spiky_ball": "R",
    "ring_toy": "R",
    "shape_sorter": "S",
    "stroller": "S",
    "tree_busy_box": "T",
    "winnie": "W",
    "winnie_cabinet": "W",
}


FURNITURE_CANNOT_ON = []
