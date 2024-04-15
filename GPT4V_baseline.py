import json
import os
import base64
import random
from functools import partial
import traceback
import time

import httpx
import openai
from tqdm import tqdm

from prompts.end2end_gpt4 import create_prompt

client = openai.OpenAI(api_key="Your Openai API key", max_retries=3,
                       timeout=httpx.Timeout(20.0, read=10.0, write=10.0, connect=3.0))


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_response(que, img):
    response = client.chat.completions.create(
        model="gpt-4-1106-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": create_prompt(que, method='all')
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        }
                    }
                ]
            }
        ],
        # timeout=httpx.Timeout(10.0, read=5.0, write=10.0, connect=5.0),
        temperature=0.2,
        max_tokens=100,
        top_p=1
    )

    return response


def convert(res):
    new_res = {}
    for qid in res:
        response = res[qid]
        try:
            ans = response.split('\n')[0]
            if ans == 'Answer:':
                ans = response.split('\n')[1]
            exp = response.split('Explanation:\n')[1]
            exp = exp.split('\n')[0].strip(' ')
            boxes = response.split('Boxes:\n')[1].strip(' ')
            boxes = boxes.split('\n')
            boxes = {b.split(': ')[0]: b.split(': ')[1] for b in boxes}

            cur_boxes = []
            while (li := exp.find('{')) >= 0:
                ri = exp.find('}')
                box_name = exp[li:ri+1]
                cur_boxes.append([[int(x) for x in boxes[box_name][1:-1].split(',')]])
                exp = exp[:li] + '[BOX]' + exp[ri+1:]

            new_res[qid] = {'answer': ans, 'explanation':exp, 'boxes':cur_boxes}
        except:
            new_res[qid] = {'answer': 'None', 'explanation': 'None', 'boxes': []}

    return new_res


if __name__ == "__main__":
    prompter = partial(create_prompt, method='all')
    questions = json.load(open("dataset/MEGQA_test.json"))
    qids = random.sample(list(questions.keys()), 3000)
    results = {}

    for qid in tqdm(qids):
        question = questions[qid]['question']
        img_id = questions[qid]['imageId']
        img_dir = f'dataset/images/{img_id}.jpg'
        image = encode_image(img_dir)

        while True:
            try:
                response = get_response(question, image)
                break
            except:
                traceback.print_exc()
                time.sleep(5)

        results[qid] = response.choices[0].message.content.strip('\n').strip()

    with open("results/GPT4_raw_results.json", 'w') as f:
        json.dump(results, f)

    converted_res = convert(results)

    with open("results/GPT4_results.json", 'w') as f:
        json.dump(converted_res, f)
