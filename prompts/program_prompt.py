import random


GQA_CURATED_EXAMPLES=[
"""Question: On which side of the photo is the old vehicle?
Program:
BOX0=LOC(image=IMAGE,object='LEFT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='old vehicle')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'left' if {ANSWER0} > 0 else 'right'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the license plate both black and small?
Program:
BOX0=LOC(image=IMAGE,object='license plate')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the license plate?')
ANSWER1=VQA(image=IMAGE0,question='How big is the license plate?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == 'black' and {ANSWER1} == 'small' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is the cup on the left side or on the right?
Program:
BOX0=LOC(image=IMAGE,object='LEFT')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='cup')
ANSWER0=COUNT(box=BOX1)
ANSWER1=EVAL(expr="'left' if {ANSWER0} > 0 else 'right'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Which color do you think the floor has?
Program:
BOX0=LOC(image=IMAGE,object='floor')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Which color do you think the floor has?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Are there either glasses or ties in the image?
Program:
BOX0=LOC(image=IMAGE,object='glasses')
BOX1=LOC(image=IMAGE,object='ties')
ANSWER0=COUNT(box=BOX0)
ANSWER1=COUNT(box=BOX1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} + {ANSWER1} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Does the woman wear gloves?
Program:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Does the woman wear gloves?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Do you see either a bench or a skateboard in the picture?
Program:
BOX0=LOC(image=IMAGE,object='bench')
BOX1=LOC(image=IMAGE,object='skateboard')
ANSWER0=COUNT(box=BOX0)
ANSWER1=COUNT(box=BOX1)
ANSWER2=EVAL(expr="'yes' if {ANSWER0} + {ANSWER1} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Who is wearing shorts?
Program:
BOX0=LOC(image=IMAGE,object='shorts')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='Who is wearing shorts?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Is the color of the table different than the color of the floor?
Program:
BOX0=LOC(image=IMAGE,object='table')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='floor')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0, question='What color is the table?')
ANSWER1=VQA(image=IMAGE1, question='What color is the floor?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} != {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is the chair that is not full white or black?
Program:
BOX0=LOC(image=IMAGE,object='chair that is not full')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the chair?')
ANSWER1=EVAL(expr="'white' if {ANSWER0} == 'white' else 'black'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: Is the baseball mitt made of leather black and open?
Program:
BOX0=LOC(image=IMAGE,object='baseball mitt made of leather')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What color is the baseball mitt?')
ANSWER1=VQA(image=IMAGE0,question='Is the baseball mitt open?')
ANSWER2=EVAL(expr="'yes' if {ANSWER1} == 'black' and {ANSWER2} == 'yes' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
"""Question: Is the man that is to the right of the woman wearing a suit?
Program:
BOX0=LOC(image=IMAGE,object='woman')
IMAGE0=CROP_RIGHTOF(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='man')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
ANSWER0=VQA(image=IMAGE1,question='What is the man wearing?')
ANSWER1=EVAL(expr="'yes' if {ANSWER0} == 'suit' else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What is the bathtub made of?
Program:
BOX0=LOC(image=IMAGE,object='bathtub')
IMAGE0=CROP(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What is the bathtub made of?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Does the man below the frame sit on a couch?
Program:
BOX0=LOC(image=IMAGE,object='frame')
IMAGE0=CROP_BELOW(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE0,object='man')
IMAGE1=CROP(image=IMAGE0,box=BOX1)
BOX2=LOC(image=IMAGE1,object='couch')
ANSWER0=COUNT(box=BOX2)
ANSWER1=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")
FINAL_RESULT=RESULT(var=ANSWER1)
""",
"""Question: What animal is to the left of the keyboard?
Program:
BOX0=LOC(image=IMAGE,object='keyboard')
IMAGE0=CROP_LEFTOF(image=IMAGE,box=BOX0)
ANSWER0=VQA(image=IMAGE0,question='What animal is in the image?')
FINAL_RESULT=RESULT(var=ANSWER0)
""",
"""Question: Are the glass and the picture frame made of the same material?
Program:
BOX0=LOC(image=IMAGE,object='glass')
IMAGE0=CROP(image=IMAGE,box=BOX0)
BOX1=LOC(image=IMAGE,object='picture frame')
IMAGE1=CROP(image=IMAGE,box=BOX1)
ANSWER0=VQA(image=IMAGE0,question='What material is the glass made of?')
ANSWER1=VQA(image=IMAGE1,question='What material is the picture frame made of?')
ANSWER2=EVAL(expr="'yes' if {ANSWER0} == {ANSWER1} else 'no'")
FINAL_RESULT=RESULT(var=ANSWER2)
""",
]


def create_prompt(inputs, num_prompts=8, method='random', seed=42):
    if method == 'all':
        prompt_examples = GQA_CURATED_EXAMPLES
    elif method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(GQA_CURATED_EXAMPLES, num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Think step by step to answer the question.\n\n{prompt_examples}'

    return prompt_examples + "\nQuestion: {question}\nProgram:".format(**inputs)
