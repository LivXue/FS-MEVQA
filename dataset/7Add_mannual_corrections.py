import argparse
import os
import json


parser = argparse.ArgumentParser(description="Correct grammar by GPT")
parser.add_argument("--data", type=str, default='./processed_data', help="path to preprocessed data")
parser.add_argument("--save", type=str, default='./processed_data', help="path for saving the data")
args = parser.parse_args()


for split in ['val']:  # 'train',
    explanation = json.load(open(os.path.join(args.data, 'explanation_' + split + '.json')))
    corrections = json.load(open(os.path.join(args.data, 'bad_results_' + split + '.json')))
    add_n = 0
    delet_k = []
    for k in corrections:
        ori = corrections[k]['exp']
        res = corrections[k]['res']

        if len(res.split(' ')) > len(ori.split(' ')) + 10:
            continue

        # check [BOX] number
        nt_box = 0
        ind = [-1]
        while (cur_ind := res[ind[-1] + 1:].find('[BOX]')) != -1:
            ind.append(cur_ind + ind[-1] + 1)
            nt_box += 1

        nd_box = 0
        for obj in corrections[k]['box']:
            nd_box += len(corrections[k]['box'][obj])

        if nt_box == nd_box:
            print(k)
            print(corrections[k])
            delet_k.append(k)
            add_n += 1

    check = input("Do you want to save these changes? [y/n]")

    if check == 'y' or check == 'yes':
        for k in delet_k:
            explanation[k] = {'exp': corrections[k]['res'], 'box': corrections[k]['box']}
            del corrections[k]

        with open(os.path.join(args.save, 'explanation_' + split + '.json'), 'w') as f:
            json.dump(explanation, f)

        with open(os.path.join(args.save, 'bad_results_' + split + '.json'), 'w') as f:
            json.dump(corrections, f)

        print("Added {} explanations to {} set".format(add_n, split))
