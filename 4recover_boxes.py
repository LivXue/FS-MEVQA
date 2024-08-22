import os
#import PIL.Image
#from PIL import Image
import json
import pickle

import torch
from tqdm import tqdm

from engine.step_interpreters import LocInterpreter

os.environ['CUDA_VISIBLE_DEVICES'] = "6"


if __name__ == '__main__':
    processes = json.load(open("dataset/processed_data/processes.json"))
    questions = json.load(open("dataset/SME_test.json"))
    explanation = json.load(open("results/generated_explanations.json"))
    qids = list(explanation.keys())

    results = {}
    complettion = True

    if complettion:
        locator = LocInterpreter()

    for k in tqdm(qids):
        if k not in processes:
            results[k] = {'answer': 'None', 'explanation': 'None', 'boxes': []}
            continue

        exp = explanation[k]
        state = pickle.load(open("dataset/states/{}.pkl".format(k), 'rb'))
        cur_boxes = []

        while (li := exp.find('{')) != -1:
            ri = exp.find('}')
            assert li < ri

            box_name = exp[li+1:ri]
            if box_name.startswith('BOX') and box_name in state:
                if isinstance(state[box_name], list):
                    cur_boxes.append(state[box_name])
                    exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                elif complettion:
                    obj_name = exp[exp[:li-1].rfind(' ')+1:li-1]
                    cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
                    exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                else:
                    exp = exp[:li-1] + exp[ri + 1:]
                    #cur_boxes.append([])
                    #print(f"{k}: {box_name}")
                #exp = exp[:li] + '[BOX]' + exp[ri+1:]
            elif box_name.startswith('IMAGE'):
                if box_name == 'IMAGE':
                    cur_boxes.append([[0, 0, *state['IMAGE'].size]])
                    exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                else:
                    box_name = 'BOX' + box_name[5:]
                    if isinstance(state[box_name], list):
                        cur_boxes.append(state[box_name])
                        exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                    elif complettion:
                        obj_name = exp[exp[:li - 1].rfind(' ') + 1:li - 1]
                        cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
                        exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                    else:
                        exp = exp[:li-1] + exp[ri + 1:]
                        #cur_boxes.append([])
                        #print(f"{k}: {box_name}")
                    #exp = exp[:li] + '[BOX]' + exp[ri + 1:]
            elif complettion:
                obj_name = exp[exp[:li - 1].rfind(' ') + 1:li - 1]
                print("Add objective name: {}".format(obj_name))
                cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
                exp = exp[:li] + '[BOX]' + exp[ri + 1:]
            else:
                print("Unknown box name: {}".format(box_name))
                #cur_boxes.append([])
                #exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                exp = exp[:li - 1] + exp[ri + 1:]

        results[k] = {'answer': processes[k]['predicted_answer'], 'explanation': exp, 'boxes': cur_boxes}

    with open("results/MEAgent_results.json", "w") as f:
        json.dump(results, f)
