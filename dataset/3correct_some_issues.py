import json
import os
import time
import re
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Correct some language issues")
parser.add_argument("--data", type=str, default='./data', help="path to preprocessed data")
parser.add_argument("--question", type=str, default='./data', help="path to GQA questions and scene graphs")
parser.add_argument("--save", type=str, default='./data', help="path for saving the data")
args = parser.parse_args()


def correct_pos(cur_exp, cur_type, answer):
    # fix reversed description about position (type 1: left/right, type 2: behind/above/below, type 3: in front of/on top of)
    reverse_dict = {'is to the left of': 'is to the right of', 'is to the right of': 'is to the left of',
                    'is behind': 'is in front of',
                    'is above': 'is below', 'is below': 'is above', 'is in front of': 'is behind',
                    'is on top of': 'is below'}

    offset_dict = {1: 5, 2: 2, 3: 4}  # length of description about relative positions
    offset = offset_dict[cur_type]

    tmp_idx = 0
    count_obj = 0
    flag = True
    for i in range(len(cur_exp)):
        if 'obj:' in cur_exp[i] or 'obj*:' in cur_exp[i]:
            count_obj += 1  # count the number of grounded objects
        if cur_exp[i] == 'that' and flag:
            tmp_idx = i  # find the first "that" as the pivot for reversing the phrases
            flag = False

    if count_obj == 2:
        if answer == 'yes':
            # in this case, we need to reverse the order of grounded objects to make it consistent with their referal order in the question
            cur_exp = cur_exp[tmp_idx + 1:] + cur_exp[:tmp_idx]
        else:
            # in this case, we need to reverse the relative locations grounded objects
            pos = ' '.join(cur_exp[-offset:])
            reverse_pos = reverse_dict[pos].split(' ')
            cur_exp = cur_exp[:tmp_idx] + reverse_pos + cur_exp[tmp_idx + 1:-offset]
    else:
        if answer in ['yes', 'no']:
            # in this case, we need to reverse the relative locations of grounded objects
            pos = ' '.join(cur_exp[-offset:])
            reverse_pos = reverse_dict[pos].split(' ')
            cur_exp = cur_exp[:tmp_idx] + reverse_pos + cur_exp[tmp_idx + 1:-offset]
        else:
            # in this case, we simply add an "is" after the first grounded object (appeared at location 1)
            cur_exp = [cur_exp[0]] + ['is'] + cur_exp[1:]
    return cur_exp


if __name__ == '__main__':
    special_token = {'mice': 'mouse'}

    rel_pool = []
    for split in ['train', 'val']:
        explanation = json.load(open(os.path.join(args.data, 'processed_explanation_' + split + '.json')))
        question = json.load(open(os.path.join(args.question, split + '_balanced_questions.json')))
        processed_data = dict()
        for idx, qid in tqdm(enumerate(explanation)):
            cur_exp = explanation[qid].replace('?', '')
            cur_exp = [cur for cur in cur_exp.split(' ') if cur not in ['', ' ']]
            for cur_word in cur_exp:
                if cur_word in special_token:
                    cur_word = special_token[cur_word]
            cur_exp = ' '.join(cur_exp)
            if 'ERROR' in cur_exp or 'obj*:scene' in cur_exp or cur_exp == '':
                # processed_data[qid] = ''
                continue

            cur_exp = cur_exp.replace(') and (', '), (')
            # fix missing 'is'
            if 'to the left of' in cur_exp and not ' is ' in cur_exp:
                cur_exp = cur_exp.split('to the left of')[0] + 'is ' + 'to the left of' + \
                          cur_exp.split('to the left of')[1]
            elif 'to the right of' in cur_exp and not ' is ' in cur_exp:
                cur_exp = cur_exp.split('to the right of')[0] + 'is ' + 'to the right of' + \
                          cur_exp.split('to the right of')[1]

            # fix adjective related to position
            pos_pool = {'is left': 'is on the left', 'is right': 'is on the right', 'is top': 'is at the top',
                        'is bottom': 'is at the bottom'}
            for cur_pos in pos_pool:
                cur_exp = cur_exp.replace(cur_pos, pos_pool[cur_pos])

            cur_exp = cur_exp.split(' ')

            # fix reversed description about position
            if ' '.join(cur_exp[-5:]) in ['is to the left of', 'is to the right of']:
                cur_exp = correct_pos(cur_exp, 1, question[qid]['answer'])
            elif ' '.join(cur_exp[-2:]) in ['is behind', 'is above', 'is below']:
                cur_exp = correct_pos(cur_exp, 2, question[qid]['answer'])
            elif ' '.join(cur_exp[-4:]) in ['is in front of', 'is on top of']:
                cur_exp = correct_pos(cur_exp, 3, question[qid]['answer'])

            # fix reversed description about relationship (e.g., xxx that xxx is wearing)
            if cur_exp[-1][-3:] == 'ing' and cur_exp[-1] != 'building' and cur_exp[-2] == 'is':
                if cur_exp[-1] not in rel_pool:
                    rel_pool.append(cur_exp[-1])
                tmp_idx = 0
                count_obj = 0
                flag = True
                for i in range(len(cur_exp)):
                    if 'obj:' in cur_exp[i] or 'obj*:' in cur_exp[i]:
                        count_obj += 1  # count the number of grounded objects
                    if cur_exp[i] == 'that' and flag:
                        tmp_idx = i  # find the first "that" as the pivot for reversing the phrases
                        flag = False

                if count_obj == 2 and 'that' in cur_exp:
                    cur_exp = cur_exp[tmp_idx + 1:] + cur_exp[:tmp_idx]

            cur_exp = ' '.join(cur_exp)
            # fix issues related to not(xxx)
            if 'not(' in cur_exp:
                not_attr = re.findall(r'not\(([^)]+)\)', cur_exp)
                if len(not_attr) == 1:  # only 0.003% samples have more than 1 "not(" attribute thus ignore them
                    not_attr = not_attr[0]
                    # find the grounded object corresponding to the attribute
                    cur_obj = cur_exp.split('not(' + not_attr + ') ')[-1].split(' ')[0]

                    # remove 'not(' and add the corresponding description at the end of explanation
                    cur_exp = cur_exp.replace('not(' + not_attr + ')', '')
                    if not 'is' in cur_exp:
                        cur_exp = 'there is ' + cur_exp
                    cur_exp = cur_exp + ', and ' + cur_obj + ' is not ' + not_attr

            processed_data[qid] = ' '.join([cur for cur in cur_exp.split(' ') if cur not in ['', ' ']])

        with open(os.path.join(args.data, 'processed_explanation_' + split + '2.json'), 'w') as f:
            json.dump(processed_data, f)

        print("Saved {} {} samples!".format(len(processed_data), split))