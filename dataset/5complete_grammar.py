import json
import os
import time
import argparse

from tqdm import tqdm


parser = argparse.ArgumentParser(description="Complete the language grammar")
parser.add_argument("--data", type=str, default='./data', help="path to preprocessed data")
parser.add_argument("--save", type=str, default='./processed_data', help="path for saving the data")
args = parser.parse_args()


all_colors = ['dark', 'brown', 'blond', 'blue', 'green', 'yellow', 'black', 'gray', 'white', 'red', 'purple',
                  'light brown', 'light blue', 'silver', 'pink', 'orange', 'tan', 'dark brown', 'beige', 'gold',
                  'cream colored', 'maroon', 'dark blue', 'khaki', 'teal', 'brunette']
all_directions = ['left', 'right', 'middle', 'bottom', 'top']
all_sizes = ['large', 'small', 'huge', 'little', 'giant', 'tiny', 'thick']
all_ages = ['little', 'young', 'old', 'adult']
all_lengths = ['long', 'short']
all_heights = ['tall', 'short']
all_material = ['metal', 'wood', 'concrete', 'plastic', 'brick', 'glass', 'leather', 'denim', 'wool', 'wire', 'rubber'
                'cement']
all_shapes = ['square', 'round', 'rectangular']
all_poses = ['lying', 'sitting', 'walking', 'standing', 'grazing', 'resting']

all_attribute = [*all_colors, *all_directions, *all_sizes, *all_ages, *all_lengths, *all_heights, *all_material,
                 *all_shapes]

unify_words = {'cooky': 'cookie', 'eye glasses': 'eyeglasses', 'ear bud': 'earbud', 'tea pot': 'teapot',
               'street light': 'streetlight', 'bird cage': 'birdcage'}
plural_shoes = ['shoes', 'boots', 'sandals', 'sneakers']

for split in ['train', 'val']:
    explanation = json.load(open(os.path.join(args.data, 'converted_explanation_' + split + '.json')))
    explanation = ' ' + explanation
    for qid in tqdm(explanation):
        exp = explanation[qid]['exp']
        for attr in all_attribute:
            exp = exp.replace(' ' + attr + ' the ', ' the ' + attr + ' ')
        exp = exp.replace('not the ', 'the not ')

        # a strange error in GQA
        exp = exp.replace(' stadning ', ' standing ')

        for k, v in unify_words.items():
            exp = exp.replace(' ' + k, ' ' + v)
            if k in explanation[qid]['box']:
                explanation[qid]['box'][v] = explanation[qid]['box'][k]
                del explanation[qid]['box'][k]

        for k in plural_shoes:
            exp = exp.replace(' is ' + k, ' are ' + k)

        exp = exp.strip()
        exp = exp[0].upper() + exp[1:] + '.'
        explanation[qid]['exp'] = exp

    with open(os.path.join(args.save, 'converted_explanation_' + split + '.json'), 'w') as f:
        json.dump(explanation, f)
