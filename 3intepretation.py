import os
from functools import partial
import traceback
import time

#import PIL.Image
#from PIL import Image
import json
import pickle
from tqdm import tqdm

from engine.utils import ProgramGenerator
from prompts.gqa_interpretation import create_prompt

os.environ['OPENAI_API_KEY'] = 'Your OpenAI API key'


if __name__ == '__main__':
    processes = json.load(open("dataset/processed_data/processes.json"))
    questions = json.load(open("dataset/MEGQA_test.json"))
    prompter = partial(create_prompt, method='all')
    generator = ProgramGenerator(prompter=prompter)
    explanation = {}
    qids = list(processes.keys())
    for k in tqdm(qids):
        p = processes[k]
        q = p['question']
        #img_id = p['image_id']
        try:
            process = p['process']
        except:
            explanation[k] = ''
            continue
        question = questions[k]

        while True:
            try:
                exp, _ = generator.generate(dict(question=question, program=process))
                break
            except:
                traceback.print_exc()
                time.sleep(5)

        explanation[k] = exp

    with open("generated_explanations.json", "w") as f:
        json.dump(explanation, f)
