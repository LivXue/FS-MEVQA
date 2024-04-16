import io, tokenize

import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import (OwlViTProcessor, OwlViTForObjectDetection,
                          MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
                          CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)

from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks


def parse_step(step_str, partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',', '=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2 * i].string] = arg_tokens[2 * i + 1].string
    parsed_result['args'] = args
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'


def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'


class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert (step_name == self.step_name)
        return step_input, output_var

    def html(self, eval_expression, step_input, step_output, output_var):
        eval_expression = eval_expression.replace('{', '').replace('}', '')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def process(self, eval_expression, step_input, step_output, output_var):
        # Non-html execution process
        eval_expression = eval_expression.replace('{', '').replace('}', '')
        return f"""{output_var}={self.step_name}(expression="{eval_expression}")={self.step_name}(expression="{step_input}")={step_output}"""

    def execute(self, prog_step, inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name, var_value in prog_step.state.items():
            if isinstance(var_value, str):
                if var_value in ['yes', 'no']:
                    prog_state[var_name] = var_value == 'yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            else:
                prog_state[var_name] = var_value

        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor', '!=')

        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(eval_expression, step_input, step_output, output_var)
            return step_output, process_str

        return step_output


class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert (step_name == self.step_name)
        return output_var

    def html(self, output, output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output, 300)
        else:
            output = html_output(output)

        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def process(self, output, output_var):
        # Non-html execution process
        return f"""{self.step_name} -> {output_var} -> {output}"""

    def execute(self, prog_step, inspect=False):
        output_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            html_str = self.html(output, output_var)
            return output, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output, output_var)
            return output, process_str

        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {self.device}')
        self.processor = AutoProcessor.from_pretrained("./Salesforce/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "./Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.model.eval()

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return img_var, question, output_var

    def predict(self, img, question):
        encoding = self.processor(img, question, return_tensors='pt')
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def html(self, img, question, answer, output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        answer = html_output(answer)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        question_arg = html_arg_name('question')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;{question_arg}='{question}')={answer}</div>"""

    def process(self, img_var, question, answer, output_var):
        # Non-html execution process
        return f"""{output_var}={self.step_name}(image={{{img_var}}}, question='{question}')={answer}"""

    def execute(self, prog_step, inspect=False):
        img_var, question, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        answer = self.predict(img, question)
        prog_step.state[output_var] = answer
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            html_str = self.html(img, question, answer, output_var)
            return answer, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(img_var, question, answer, output_var)
            return answer, process_str

        return answer


class SizeInterpreter():
    step_name = 'SIZE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        box = args['box']
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return box, output_var

    def predict(self, box):
        size = (box[2] - box[0]) * (box[3] - box[1])

        if size > 480 * 480:
            return "large"
        elif size > 320 * 320:
            return "medium"
        else:
            return "small"

    def html(self, box, size, output_var):
        step_name = html_step_name(self.step_name)
        size_str = html_output(size)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('box')
        box_str = html_var_name(box)
        return f"""<div>{output_var}={step_name}({box_arg}={box_str})={size_str}</div>"""

    def process(self, box, size, output_var):
        # Non-html execution process
        return f"""{output_var}={self.step_name}(box={{{box}}})={size}"""

    def execute(self, prog_step, inspect=False):
        box_name, output_var = self.parse(prog_step)
        box = prog_step.state[box_name][0]
        size = self.predict(box)
        prog_step.state[output_var] = size
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            html_str = self.html(box_name, size, output_var)
            return size, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(box_name, size, output_var)
            return size, process_str

        return size


class LocInterpreter():
    step_name = 'LOC'

    def __init__(self, thresh=0.1, nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {self.device}')
        self.processor = OwlViTProcessor.from_pretrained(
            "./google/owlvit-large-patch14")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "./google/owlvit-large-patch14").to(self.device)
        self.model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return img_var, obj_name, output_var

    def normalize_coord(self, bbox, img_size):
        w, h = img_size
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)
        return [x1, y1, x2, y2]

    def predict(self, img, obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']],
            images=img,
            return_tensors='pt')
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k, v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v

        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=self.thresh,
                                                               target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes) == 0:
            return []

        boxes, scores = zip(*sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i], img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes, selected_scores, self.nms_thresh)
        return selected_boxes

    def top_box(self, img):
        w, h = img.size
        return [0, 0, w - 1, int(h / 2)]

    def bottom_box(self, img):
        w, h = img.size
        return [0, int(h / 2), w - 1, h - 1]

    def left_box(self, img):
        w, h = img.size
        return [0, 0, int(w / 2), h - 1]

    def right_box(self, img):
        w, h = img.size
        return [int(w / 2), 0, w - 1, h - 1]

    def box_image(self, img, boxes, highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i, box in enumerate(boxes):
            if i == 0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box, outline=color, width=5)

        return img1

    def html(self, img, box_img, output_var, obj_name):
        step_name = html_step_name(self.step_name)
        obj_arg = html_arg_name('object')
        img_arg = html_arg_name('image')
        output_var = html_var_name(output_var)
        img = html_embed_image(img)
        box_img = html_embed_image(box_img, 300)
        return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"

    def process(self, img_var, box_img, output_var, obj_name):
        # Non-html execution process
        return f"""{output_var}={self.step_name}(image={{{img_var}}}, object='{obj_name}')={{{box_img}}}"""

    def execute(self, prog_step, inspect=False):
        img_var, obj_name, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name == 'TOP':
            bboxes = [self.top_box(img)]
        elif obj_name == 'BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name == 'LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name == 'RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img, obj_name)

        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var + '_IMAGE'] = box_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(img_var, output_var + '_IMAGE', output_var, obj_name)
            return bboxes, process_str

        return bboxes


class Loc2Interpreter(LocInterpreter):

    def execute(self, prog_step, inspect='None'):
        img_var, obj_name, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        bboxes = self.predict(img, obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))
        prog_step.state[output_var] = objs

        if inspect:
            box_img = self.box_image(img, bboxes, highlight_best=False)
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return objs


class CountInterpreter():
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return box_var, output_var

    def html(self, box_img, output_var, count):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_img = html_embed_image(box_img)
        output = html_output(count)
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={output}</div>"""

    def process(self, box_img, output_var, count):
        # Non-html execution process
        return f"""{output_var}={self.step_name}(bbox={{{box_img}}})={count}"""

    def execute(self, prog_step, inspect=False):
        box_var, output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count = len(boxes)
        prog_step.state[output_var] = count
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(box_img, output_var, count)
            return count, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(box_var + '_IMAGE', output_var, count)
            return count, process_str

        return count


class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self, box, img_size, factor=1.5):
        W, H = img_size
        x1, y1, x2, y2 = box
        dw = int(factor * (x2 - x1) / 2)
        dh = int(factor * (y2 - y1) / 2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0, cx - dw)
        x2 = min(cx + dw, W)
        y1 = max(0, cy - dh)
        y2 = min(cy + dh, H)
        return [x1, y1, x2, y2]

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return img_var, box_var, output_var

    def html(self, img, out_img, output_var, box_img):
        img = html_embed_image(img)
        out_img = html_embed_image(out_img, 300)
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_img}</div>"""

    def process(self, out_img, output_var, box_img):
        # Non-html execution process
        return f"""{output_var}={self.step_name}(bbox={{{box_img}}})={{{out_img}}}"""

    def execute(self, prog_step, inspect=False):
        img_var, box_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            box = []
            out_img = img

        prog_step.state[output_var] = out_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output_var, output_var, box_var + '_IMAGE')
            return out_img, process_str

        return out_img


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self, box, img_size):
        w, h = img_size
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        return [cx, 0, w - 1, h - 1]

    def execute(self, prog_step, inspect=False):
        img_var, box_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            right_box = self.right_of(box, img.size)
        else:
            w, h = img.size
            box = []
            right_box = [int(w / 2), 0, w - 1, h - 1]

        out_img = img.crop(right_box)

        prog_step.state[output_var] = out_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output_var, output_var, box_var + '_IMAGE')
            return out_img, process_str

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self, box, img_size):
        w, h = img_size
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        return [0, 0, cx, h - 1]

    def execute(self, prog_step, inspect=False):
        img_var, box_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            left_box = self.left_of(box, img.size)
        else:
            w, h = img.size
            box = []
            left_box = [0, 0, int(w / 2), h - 1]

        out_img = img.crop(left_box)

        prog_step.state[output_var] = out_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output_var, output_var, box_var + '_IMAGE')
            return out_img, process_str

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self, box, img_size):
        w, h = img_size
        x1, y1, x2, y2 = box
        cy = int((y1 + y2) / 2)
        return [0, 0, w - 1, cy]

    def execute(self, prog_step, inspect=False):
        img_var, box_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            above_box = self.above(box, img.size)
        else:
            w, h = img.size
            box = []
            above_box = [0, 0, int(w / 2), h - 1]

        out_img = img.crop(above_box)

        prog_step.state[output_var] = out_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output_var, output_var, box_var + '_IMAGE')
            return out_img, process_str

        return out_img


class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self, box, img_size):
        w, h = img_size
        x1, y1, x2, y2 = box
        cy = int((y1 + y2) / 2)
        return [0, cy, w - 1, h - 1]

    def execute(self, prog_step, inspect=False):
        img_var, box_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            below_box = self.below(box, img.size)
        else:
            w, h = img.size
            box = []
            below_box = [0, 0, int(w / 2), h - 1]

        out_img = img.crop(below_box)

        prog_step.state[output_var] = out_img
        if isinstance(inspect, str) and inspect.casefold() == 'html':
            box_img = prog_step.state[box_var + '_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str
        elif isinstance(inspect, str) and inspect.casefold() == 'text':
            process_str = self.process(output_var, output_var, box_var + '_IMAGE')
            return out_img, process_str

        return out_img


class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'


class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'


class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'


class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'


class SegmentInterpreter():
    step_name = 'SEG'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            "facebook/maskformer-swin-base-coco")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco").to(self.device)
        self.model.eval()

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return img_var, output_var

    def pred_seg(self, img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
        instance_map = outputs['segmentation'].cpu().numpy()
        objs = []
        print(outputs.keys())
        for seg in outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            mask = (instance_map == inst_id).astype(float)
            resized_mask = np.array(
                Image.fromarray(mask).resize(
                    img.size, resample=Image.BILINEAR))
            Y, X = np.where(resized_mask > 0.5)
            x1, x2 = np.min(X), np.max(X)
            y1, y2 = np.min(Y), np.max(Y)
            num_pixels = np.sum(mask)
            objs.append(dict(
                mask=resized_mask,
                category=category,
                box=[x1, y1, x2, y2],
                inst_id=inst_id
            ))

        return objs

    def html(self, img_var, output_var, output):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        img_arg = html_arg_name('image')
        output = html_embed_image(output, 300)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var})={output}</div>"""

    def execute(self, prog_step, inspect=False):
        img_var, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.pred_seg(img)
        prog_step.state[output_var] = objs
        if inspect:
            labels = [str(obj['inst_id']) + ':' + obj['category'] for obj in objs]
            obj_img = vis_masks(img, objs, labels)
            html_str = self.html(img_var, output_var, obj_img)
            return objs, html_str

        return objs


class SelectInterpreter():
    step_name = 'SELECT'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = eval(parse_result['args']['query']).split(',')
        category = eval(parse_result['args']['category'])
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return img_var, obj_var, query, category, output_var

    def calculate_sim(self, inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats, text_feats.t())

    def query_obj(self, query, objs, img):
        images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu().numpy()

        obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids]

    def html(self, img_var, obj_var, query, category, output_var, output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        category_arg = html_arg_name('category')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output, 300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{query_arg}={query},{category_arg}={category})={output}</div>"""

    def query_string_match(self, objs, q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q, f'{q}-merged', f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category'] == cat]

        return None

    def execute(self, prog_step, inspect=False):
        img_var, obj_var, query, category, output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        select_objs = []

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs

        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue

                select_objs += matches

        if query is not None and len(select_objs) == 0:
            select_objs = self.query_obj(query, objs, img)

        prog_step.state[output_var] = select_objs
        if inspect:
            select_obj_img = vis_masks(img, select_objs)
            html_str = self.html(img_var, obj_var, query, category, output_var, select_obj_img)
            return select_objs, html_str

        return select_objs


class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        image_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        category_var = parse_result['args']['categories']
        output_var = parse_result['output_var']
        assert (step_name == self.step_name)
        return image_var, obj_var, category_var, output_var

    def calculate_sim(self, inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats, text_feats.t())

    def query_obj(self, query, objs, img):
        if len(objs) == 0:
            images = [img]
            return []
        else:
            images = [img.crop(obj['box']) for obj in objs]

        if len(query) == 1:
            query = query + ['other']

        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            sim = self.calculate_sim(inputs)

        # if only one query then select the object with the highest score
        if len(query) == 1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class'] = query[0]
            obj['class_score'] = 100.0 * scores[obj_ids[0], 0]
            return [obj]

        # assign the highest scoring class to each object but this may assign same class to multiple objects
        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)
        for i, (obj, cat_id) in enumerate(zip(objs, cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i, cat_id]
            obj['class'] = class_name  # + f'({score_str})'
            obj['class_score'] = round(class_score * 100, 1)

        # sort by class scores and then for each class take the highest scoring object
        objs = sorted(objs, key=lambda x: x['class_score'], reverse=True)
        objs = [obj for obj in objs if 'class' in obj]
        classes = set([obj['class'] for obj in objs])
        new_objs = []
        for class_name in classes:
            cls_objs = [obj for obj in objs if obj['class'] == class_name]

            max_score = 0
            max_obj = None
            for obj in cls_objs:
                if obj['class_score'] > max_score:
                    max_obj = obj
                    max_score = obj['class_score']

            new_objs.append(max_obj)

        return new_objs

    def html(self, img_var, obj_var, objs, cat_var, output_var):
        step_name = html_step_name(self.step_name)
        output = []
        for obj in objs:
            output.append(dict(
                box=obj['box'],
                tag=obj['class'],
                score=obj['class_score']
            ))
        output = html_output(output)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        cat_var = html_var_name(cat_var)
        obj_var = html_var_name(obj_var)
        img_arg = html_arg_name('image')
        cat_arg = html_arg_name('categories')
        return f"""<div>{output_var}={step_name}({img_arg}={img_var},{cat_arg}={cat_var})={output}</div>"""

    def execute(self, prog_step, inspect=False):
        image_var, obj_var, category_var, output_var = self.parse(prog_step)
        img = prog_step.state[image_var]
        objs = prog_step.state[obj_var]
        cats = prog_step.state[category_var]
        objs = self.query_obj(cats, objs, img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(image_var, obj_var, objs, category_var, output_var)
            return objs, html_str

        return objs


def dummy(images, **kwargs):
    return images, False


def register_step_interpreters(dataset='SME'):
    if dataset == 'SME':
        return dict(
            LOC=LocInterpreter(),
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter(),
            SIZE=SizeInterpreter()
        )
    else:
        raise NotImplementedError("Unknown interpreter")
