import json
import os
import time
import traceback
import argparse

import httpx
from tqdm import tqdm
import openai

parser = argparse.ArgumentParser(description="Correct grammar by GPT")
parser.add_argument("--data", type=str, default='./processed_data', help="path to preprocessed data")
parser.add_argument("--save", type=str, default='./processed_data', help="path for saving the data")
args = parser.parse_args()

client = openai.OpenAI(api_key="Your OpenAI API Key", max_retries=3,
                       timeout=httpx.Timeout(20.0, read=10.0, write=10.0, connect=3.0))

conflict_a = {'sign': 'street sign', 'sign': 'stop sign', 'sign': 'traffic sign',  'car': 'train car',
              'bag': 'trash bag', 'bag': 'shopping bag', 'drink': 'soft drink', 'table': 'side table',
              'bear': 'teddy bear', 'table': 'coffee table', 'sauce': 'tomato sauce', 'food': 'dog food',
              'food': 'cat food'}
conflict_b = {'computer': 'computer desk', 'computer': 'computer mouse', 'train': 'train tracks', 'train': 'train car',
              'soap': 'soap bottle', 'bus': 'bus stop', 'toilet': 'toilet paper', 'pizza': 'pizza pan',
              'ski': 'ski lift', 'computer': 'computer monitor', 'train': 'train station', 'bus': 'bus driver',
              'pizza': 'pizza tray', 'coffee': 'coffee mug', 'street': 'street sign', 'flower': 'flower pot',
              'clock': 'clock tower', 'toy': 'toy car', 'baseball': 'baseball bat', 'coffee': 'coffee cup',
              'ceiling': 'ceiling light', 'wine': 'wine bottle'}

def get_response(s):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": "Please analyze the input text for grammatical errors and provide the corrected text. "
                           "Focus on the positions of \"the\" and overall clarity to ensure a polished and error-free piece. "
                           "Remember the singular and plural forms of nouns are right. Please keep the sentence "
                           "structure. Please directly output the corrected text. If there are no grammatical "
                           "errors, please repeat the original text."
            },
            {
                "role": "user",
                "content": "Input: {}\nOutput: ".format(s)
            }
        ],
        # timeout=httpx.Timeout(10.0, read=5.0, write=10.0, connect=5.0),
        temperature=0.2,
        max_tokens=100,
        top_p=1
    )

    return response


for split in ['train']:  # 'val',
    explanation = json.load(open(os.path.join(args.data, 'converted_explanation_' + split + '.json')))
    processed_explanation = dict()
    bad_results = dict()
    for n, qid in tqdm(enumerate(explanation)):
        if n % 50000 == 49999:
            with open(os.path.join(args.save, 'explanation_' + split + '.json'), 'w') as f:
                json.dump(processed_explanation, f)

            with open(os.path.join(args.save, 'bad_results_' + split + '.json'), 'w') as f:
                json.dump(bad_results, f)

        exp = explanation[qid]['exp']

        # remove [BOX]
        exp = exp.replace(' [BOX]', '')

        retry = 0
        while retry < 5:
            try:
                response = get_response(exp)
                break
            except:
                retry += 1
                traceback.print_exc()
                time.sleep(5)

        if retry == 5:
            bad_results[qid] = {'exp': explanation[qid]['exp'], 'res': '[no response]', 'box': explanation[qid]['box']}
            continue

        res = response.choices[0].message.content.strip('\n').strip()
        res = res.split('Input:')[-1].split('Output:')[-1].strip()
        if len(res.split(' ')) < len(exp.split(' ')) + 10:
            # add [BOX] back
            res = ' ' + res.lower()
            # strange errors of GPT
            res = res.replace(' shoess ', ' shoes ').replace(' shoess.', ' shoes.')
            res = res.replace(' pokémon ', ' pokemon ').replace(' pokémon.', ' pokemon.')

            for obj in explanation[qid]['box']:
                res = res.replace(' ' + obj + ' ', ' ' + obj + ' [BOX] ').replace(
                    ' ' + obj + '.', ' ' + obj + ' [BOX].').replace(
                    ' ' + obj + ',', ' ' + obj + ' [BOX],').replace(
                    '[BOX] [BOX]', '[BOX]')

            res = res.replace('[BOX] [BOX]', '[BOX]')
            res = res.replace(' located here', ' located here [BOX]')

            # important: b first and a second (consider street [BOX] sign [BOX])
            for k, v in conflict_b.items():
                res = res.replace(' ' + v.split()[0] + ' [BOX] ' + v.split()[1] + ' ', ' ' + v + ' ')
                res = res.replace(' ' + v.split()[0] + ' [BOX] ' + v.split()[1] + '.', ' ' + v + '.')

            for k, v in conflict_a.items():
                if k in explanation[qid]['box'] and v not in explanation[qid]['box']:
                    res = res.replace(' ' + v + ' [BOX]', ' ' + v)

            res = res.strip()
            res = res[0].upper() + res[1:]
            # check [BOX] number
            nt_box = 0
            ind = [-1]
            while (cur_ind := res[ind[-1] + 1:].find('[BOX]')) != -1:
                ind.append(cur_ind + ind[-1] + 1)
                nt_box += 1

            nd_box = 0
            for obj in explanation[qid]['box']:
                nd_box += len(explanation[qid]['box'][obj])

            if nt_box == nd_box:
                processed_explanation[qid] = {'exp': res, 'box': explanation[qid]['box']}
            else:
                if ' he [BOX]' in exp and ' him' in res:
                    res = res.replace(' him ', ' him [BOX] ').replace(' him.', ' him [BOX].')
                    explanation[qid]['box']['him'] = explanation[qid]['box']['he']
                    del explanation[qid]['box']['he']
                if ' she [BOX]' in exp and ' her' in res:
                    res = res.replace(' her ', ' her [BOX] ').replace(' her.', ' her [BOX].')
                    explanation[qid]['box']['her'] = explanation[qid]['box']['she']
                    del explanation[qid]['box']['she']
                if ' baked good ' in exp and ' baked goods ' in res:
                    res = res.replace(' baked goods ', ' baked goods [BOX] ')
                    explanation[qid]['box']['baked goods'] = explanation[qid]['box']['baked good']
                    del explanation[qid]['box']['baked good']
                if ' vegetable ' in exp and ' vegetables ' in res:
                    res = res.replace(' vegetables ', ' vegetables [BOX] ')
                    explanation[qid]['box']['vegetables'] = explanation[qid]['box']['vegetable']
                    del explanation[qid]['box']['vegetable']

                # check [BOX] number again
                nt_box = 0
                ind = [-1]
                while (cur_ind := res[ind[-1] + 1:].find('[BOX]')) != -1:
                    ind.append(cur_ind + ind[-1] + 1)
                    nt_box += 1

                nd_box = 0
                for obj in explanation[qid]['box']:
                    nd_box += len(explanation[qid]['box'][obj])
                if nt_box == nd_box:
                    processed_explanation[qid] = {'exp': res, 'box': explanation[qid]['box']}
                else:
                    bad_results[qid] = {'exp': explanation[qid]['exp'], 'res': res, 'box': explanation[qid]['box']}
        else:
            bad_results[qid] = {'exp': explanation[qid]['exp'], 'res': res, 'box': explanation[qid]['box']}

    with open(os.path.join(args.save, 'explanation_' + split + '.json'), 'w') as f:
        json.dump(processed_explanation, f)

    with open(os.path.join(args.save, 'bad_results_' + split + '.json'), 'w') as f:
        json.dump(bad_results, f)

    print("Saved {}/{} {} samples!".format(len(processed_explanation), len(bad_results), split))
