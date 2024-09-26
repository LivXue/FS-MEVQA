import os
from functools import partial

from PIL import Image
import cv2
from colorama import Fore

from engine.utils import ProgramGenerator, ProgramInterpreter
from prompts.explanation_prompt import create_prompt as interpretation_prompt
from prompts.program_prompt import create_prompt as program_prompt


os.environ['OPENAI_API_KEY'] = 'Your OpenAI API key'

# for colored output
colors = [(0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0)]
p_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

program_prompter = partial(program_prompt, method='all')
program_generator = ProgramGenerator(prompter=program_prompter)
interpreter = ProgramInterpreter(dataset='SME')
locator = interpreter.step_interpreters['LOC']
interpretation_prompter = partial(interpretation_prompt, method='all')
interpretation_generator = ProgramGenerator(prompter=interpretation_prompter)


def add_boxes(img, boxes, color):
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    return img


def reasoning(img_path, question):
    image = Image.open(img_path)
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    init_state = dict(
        IMAGE=image.convert('RGB')
    )

    # multimodal programming
    program, _ = program_generator.generate(dict(question=question))
    result, state, process = interpreter.execute(program, init_state, inspect='text')

    # explanation translation
    exp, _ = interpretation_generator.generate(dict(question=question, program=process))

    # post-processing (rethinking)
    cur_boxes = []

    while (li := exp.find('{')) != -1:
        ri = exp.find('}')
        assert li < ri

        box_name = exp[li + 1:ri]
        if box_name.startswith('BOX') and box_name in state:
            if isinstance(state[box_name], list):
                cur_boxes.append(state[box_name])
                exp = exp[:li] + '[BOX]' + exp[ri + 1:]
            else:
                obj_name = exp[exp[:li - 1].rfind(' ') + 1:li - 1]
                cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
                exp = exp[:li] + '[BOX]' + exp[ri + 1:]
        elif box_name.startswith('IMAGE'):
            if box_name == 'IMAGE':
                cur_boxes.append([[0, 0, *state['IMAGE'].size]])
                exp = exp[:li] + '[BOX]' + exp[ri + 1:]
            else:
                box_name = 'BOX' + box_name[5:]
                if isinstance(state[box_name], list):
                    cur_boxes.append(state[box_name])
                    exp = exp[:li] + '[BOX]' + exp[ri + 1:]
                else:
                    obj_name = exp[exp[:li - 1].rfind(' ') + 1:li - 1]
                    cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
                    exp = exp[:li] + '[BOX]' + exp[ri + 1:]
        else:
            obj_name = exp[exp[:li - 1].rfind(' ') + 1:li - 1]
            print("Add objective name: {}".format(obj_name))
            cur_boxes.append(locator.predict(state['IMAGE'], obj_name))
            exp = exp[:li] + '[BOX]' + exp[ri + 1:]

    # display
    answer = result
    img = cv2.imread(img_path)
    for i, box in enumerate(cur_boxes):
        img = add_boxes(img, box, colors[i])
        lid = exp.find(' [BOX]')
        exp = exp[:lid + 1] + p_colors[i] + '[BOX]' + Fore.BLACK + exp[lid + 6:]
    print(f"Question: {question}")
    print(f"Program: {program}")
    print(f"Process: {process}")
    print(f"Answer: {answer}")
    print("Explanation:", exp)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    return state, program, process, cur_boxes, exp


if __name__ == '__main__':
    img_path = 'OOD_images/chart.png'
    question = 'On which side of this chart is the title?'

    state, program, process, boxes, exp = reasoning(img_path, question)
