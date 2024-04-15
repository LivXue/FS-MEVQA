import os
from PIL import Image
from functools import partial
import argparse
import json
import pickle
import traceback

from tqdm import tqdm

from engine.utils import ProgramInterpreter

parser = argparse.ArgumentParser(description='Generate programs')
parser.add_argument('--input_dir', type=str, default='dataset/processed_data/test_programs.json', help='input data directory')
parser.add_argument('--output_dir', type=str, default='dataset/processed_data/processes.json', help='output data directory')
parser.add_argument('--image_dir', type=str, default='dataset/images/', help='input image directory')
parser.add_argument('--state_dir', type=str, default='dataset/states/', help='output state directory')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "7"


if __name__ == '__main__':
    interpreter = ProgramInterpreter(dataset='MEVQA')

    data = json.load(open(args.input_dir))

    processes = {}
    for sampled_id in tqdm(data.keys()):
        sampled_data = data[sampled_id]
        question = sampled_data['question']
        image_id = sampled_data['image_id']
        program = sampled_data['program']

        try:
            image = Image.open(args.image_dir + image_id + '.jpg')
        except FileNotFoundError:
            try:
                image = Image.open(args.image_dir + image_id + '.png')
            except:
                traceback.print_exc()
                continue

        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )
        try:
            result, prog_state, pro_str = interpreter.execute(program, init_state, inspect='text')
        except:
            continue
        processes[sampled_id] = {'question': question, 'image_id': image_id, 'program': program,
                                 'answer': sampled_data['answer'], 'process': pro_str, 'predicted_answer': str(result)}
        with open(args.state_dir + sampled_id + ".pkl", 'wb') as f:
            pickle.dump(prog_state, f)

    with open(args.output_dir, 'w') as f:
        json.dump(processes, f)
