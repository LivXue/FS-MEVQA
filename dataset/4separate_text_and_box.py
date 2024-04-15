import json
import numpy as np
import re
import os
import time
import argparse
from copy import deepcopy

from tqdm import tqdm
from pattern.en import pluralize
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


wnl = WordNetLemmatizer()

parser = argparse.ArgumentParser(description="Separate text and boxes")
parser.add_argument("--data", type=str, default='./data', help="path to preprocessed data")
parser.add_argument("--question", type=str, default='./data', help="path to GQA questions and scene graphs")
parser.add_argument("--save", type=str, default='./data', help="path for saving the data")
args = parser.parse_args()


always_plural = ['trousers', 'pants', 'shorts', 'jeans', 'briefs', 'pyjamas', 'spectacles', 'glasses', 'binoculars',
                 'goggles', 'scissors', 'pincers', 'sunglasses', 'headphones']
always_singular = ['people']


def safe_pluralize(w):
    if w not in always_singular and w not in always_plural:
        return pluralize(wnl.lemmatize(w, pos='n'))
    else:
        return w


def find_all_charactor(s, c):
    i0 = s.find(c)
    if i0 == -1:
        return []
    else:
        inds = [i0]

    while (i := s[inds[-1] + 1:].find(c)) != -1:
        inds.append(i + inds[-1] + 1)

    return inds


def get_box(cur_sg, cur_obj_id):
    xyxy = [cur_sg[cur_obj_id]['x'],
            cur_sg[cur_obj_id]['y'],
            cur_sg[cur_obj_id]['x'] + cur_sg[cur_obj_id]['w'],
            cur_sg[cur_obj_id]['y'] + cur_sg[cur_obj_id]['h']]
    return xyxy


for split in ['train', 'val']:
    explanation = json.load(open(os.path.join(args.data, 'processed_explanation_' + split + '2.json')))
    question = json.load(open(os.path.join(args.question, split + '_balanced_questions.json')))
    scene_graph = json.load(open(os.path.join('sceneGraphs', split + '_sceneGraphs.json')))

    # converting textual explanation to multi-modal explanation
    converted_explanation = dict()
    for qid in tqdm(explanation):
        cur_exp = explanation[qid]
        cur_exp = cur_exp.replace('obj*:', '').replace('_', 'object')
        cur_box = dict()
        cur_obj = None

        img_id = question[qid]['imageId']
        cur_sg = scene_graph[img_id]['objects']

        # find grounded objects
        ground_obj_ids = re.findall(r'\(([^)]+)\)', cur_exp)
        ground_obj_inds = find_all_charactor(cur_exp, '(')
        filtered_obj = []
        for i in range(len(ground_obj_ids)):
            id = ground_obj_ids[i]
            if ',' in id:
                cur_obj = 'LOC'
                if 'LOC' not in cur_box:
                    cur_box['LOC'] = [[[int(x) for x in id.split(',')]]]
                else:
                    cur_box['LOC'].append([[int(x) for x in id.split(',')]])
            else:
                obj_ind = ground_obj_inds[i]
                if i == 0 or cur_exp[obj_ind-3:obj_ind] != '), ':   # first id of an obj
                    if cur_exp[obj_ind-2:obj_ind] != ', ':
                        cur_obj = cur_exp[cur_exp[:obj_ind].rfind('obj:')+4: obj_ind-1]
                        last_word = cur_obj.split(' ')[-1]
                        right_bra = cur_exp[obj_ind:].find(')') + obj_ind
                        if last_word in always_plural or (right_bra + 3 < len(cur_exp) and cur_exp[right_bra + 1:right_bra + 4] == ', ('):
                            cur_obj = cur_obj[:-len(last_word)] + safe_pluralize(last_word)
                            cur_exp = cur_exp[:cur_exp[:obj_ind].rfind('obj:') + 4] + '@' + cur_obj + ' ' + cur_exp[obj_ind:]
                        else:
                            cur_obj = cur_obj[:-len(last_word)] + wnl.lemmatize(last_word, pos='n')
                            cur_exp = cur_exp[:cur_exp[:obj_ind].rfind('obj:') + 4] + cur_obj + ' ' + cur_exp[obj_ind:]

                        ground_obj_inds = find_all_charactor(cur_exp, '(')
                        obj_ind = ground_obj_inds[i]
                    else:
                        last_word = cur_obj.split(' ')[-1]
                        right_bra = cur_exp[obj_ind:].find(')') + obj_ind
                        if last_word in always_plural or (right_bra + 3 < len(cur_exp) and cur_exp[right_bra + 1:right_bra + 4] == ', ('):
                            cur_obj = cur_obj[:-len(last_word)] + safe_pluralize(last_word)
                            cur_exp = cur_exp[:obj_ind] + 'obj:@' + cur_obj + ' ' + cur_exp[obj_ind:]
                        else:
                            cur_obj = cur_obj[:-len(last_word)] + wnl.lemmatize(last_word, pos='n')
                            cur_exp = cur_exp[:obj_ind] + 'obj:' + cur_obj + ' ' + cur_exp[obj_ind:]

                        ground_obj_inds = find_all_charactor(cur_exp, '(')
                        obj_ind = ground_obj_inds[i]

                    if cur_obj not in cur_box:
                        cur_box[cur_obj] = []
                    cur_box[cur_obj].append([get_box(cur_sg, id)])  # create new obj group

                    # add [BOX]
                    cur_exp = cur_exp[:obj_ind] + '[BOX] ' + cur_exp[obj_ind:]
                    ground_obj_inds = find_all_charactor(cur_exp, '(')
                    obj_ind = ground_obj_inds[i]
                else:
                    cur_box[cur_obj][-1].append(get_box(cur_sg, id)) # add box to an existing obj group



        # clean 'obj:'
        cur_exp = cur_exp.replace('obj:@', 'the ')
        cur_exp = cur_exp.replace('obj:', 'the ')

        cur_exp = cur_exp.replace('the she ', 'she ')
        cur_exp = cur_exp.replace('the he ', 'he ')

        # clean '(#####)'
        ground_obj_inds = find_all_charactor(cur_exp, '(')
        for i in range(len(ground_obj_inds)):
            ind = ground_obj_inds[i]
            if cur_exp[ind - 3:ind] == '), ':
                cur_exp = cur_exp[:ind - 2] + cur_exp[ind:]
                ground_obj_inds = find_all_charactor(cur_exp, '(')

        for id in ground_obj_ids:
            cur_exp = cur_exp.replace('(' + id + ')', '')

        # check negative boxes caused by annotation errors in GQA
        invalid = False
        for k in cur_box:
            groups = cur_box[k]
            for group in groups:
                for box in group:
                    for axis in box:
                        if axis < 0:
                            print("Invalid box {}!".format(box))
                            invalid = True
        if invalid:
            continue

        # remove ''
        processed_exp = [cur for cur in cur_exp.split(' ') if cur not in ['', ' ']]
        processed_exp = ' '.join(processed_exp)
        converted_explanation[qid] = {'exp': processed_exp, 'box': cur_box}

    with open(os.path.join(args.save, 'converted_explanation_' + split + '.json'), 'w') as f:
        json.dump(converted_explanation, f)

    print("Saved {} {} samples!".format(len(converted_explanation), split))
