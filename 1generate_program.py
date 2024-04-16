import os
import time
import traceback
from PIL import Image
from functools import partial
import argparse
import json

from tqdm import tqdm

from engine.utils import ProgramGenerator
from prompts.program_prompt import create_prompt

parser = argparse.ArgumentParser(description='Generate programs')
parser.add_argument('--data_dir', type=str, default='dataset/data/SME_test.json', help='input data directory')
parser.add_argument('--input_image_dir', type=str, default='../images/', help='input image directory')
parser.add_argument('--output_dir', type=str, default='dataset/processed_data/test_programs.json', help='output data directory')
parser.add_argument('--output_image_dir', type=str, default='dataset/images/', help='output image directory')
parser.add_argument('--sampling_num', type=int, default=500, help='sampling number')

args = parser.parse_args()
os.environ['OPENAI_API_KEY'] = 'Your OpenAI API KEY'


if __name__ == '__main__':
    prompter = partial(create_prompt, method='all')
    generator = ProgramGenerator(prompter=prompter)

    all_data = json.load(open(args.data_dir))
    sampled_ids = list(all_data.keys())

    output_data = {}
    for sampled_id in tqdm(sampled_ids):
        sampled_data = all_data[sampled_id]
        question = sampled_data['question']
        image_id = sampled_data['imageId']

        try:
            image = Image.open(args.input_image_dir + image_id + '.jpg')
        except FileNotFoundError:
            try:
                image = Image.open(args.input_image_dir + image_id + '.png')
            except:
                continue

        while True:
            try:
                prog, _ = generator.generate(dict(question=question))
                break
            except:
                traceback.print_exc()
                time.sleep(5)
        output_data[sampled_id] = {'question': question, 'image_id': image_id, 'program': prog, 'answer': sampled_data['answer']}
        image.save(args.output_image_dir + image_id + '.jpg')

    with open(args.output_dir, 'w') as f:
        json.dump(output_data, f)
