import os
import json
import random

from metrics import lang_eval, vis_eval, attr_evaluater



class evaluator():
    def __init__(self, gts):
        self.exp_gts = gts
        self.attr_evaluater = attr_evaluater(gts)

    def evaluate(self, exp_res):
        scores = lang_eval(self.exp_gts, exp_res)
        scores['IoU'] = vis_eval(self.exp_gts, exp_res)
        scores.update(self.attr_evaluater.score(exp_res))

        return scores


if __name__ == '__main__':
    res = json.load(open("results/MEAgent_results.json"))

    gts = json.load(open("dataset/MEGQA_test.json"))
    gts = {k: gts[k] for k in res}
    assert res.keys() == gts.keys()

    evalor = evaluator(gts)

    acc = []
    for k in res.keys():
        if res[k]['answer'] == gts[k]['answer']:
            acc.append(1)
        else:
            acc.append(0)

    acc = sum(acc) / len(acc)
    print(f"Accuracy = {acc * 100}")

    scores = evalor.evaluate(res)
    for k in scores:
        print(f"{k} = {scores[k]}")
