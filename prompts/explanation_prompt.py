import random

GQA_CURATED_EXAMPLES=[
# 1
"""Question: On which side of the photo is the old vehicle?
Program:
BOX0=LOC(image={IMAGE}, object='old vehicle')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE0}, object='LEFT')={BOX1_IMAGE}
ANSWER0=COUNT(bbox={BOX1_IMAGE})=1
ANSWER1=EVAL(expression="'left' if ANSWER0 > 0 else 'right'")=EVAL(expression="'left' if 1 > 0 else 'right'")=left
RESULT -> ANSWER1 -> left
Explanation:
The old vehicle {BOX0} is located here {BOX0}.
""",
# 2
"""Question: Is the license plate both black and small?
Program:
BOX0=LOC(image={IMAGE}, object='license plate')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='What color is the license plate?')=black
ANSWER1=VQA(image={IMAGE0}, question='How big is the license plate?')=small
ANSWER2=EVAL(expression="'yes' if ANSWER0 == 'black' and ANSWER1 == 'small' else 'no'")=EVAL(expression="'yes' if 'black' == 'black' and 'small' == 'small' else 'no'")=yes
RESULT -> ANSWER2 -> yes
Explanation:
There is a black and small license plate {BOX0}.
""",
# 3
# """Question: Are there any red kites or flags?
# Program:
# BOX0=LOC(image={IMAGE}, object='kite')={BOX0_IMAGE}
# BOX1=LOC(image={IMAGE}, object='flag')={BOX1_IMAGE}
# ANSWER0=COUNT(bbox={BOX0_IMAGE})=0
# ANSWER1=COUNT(bbox={BOX1_IMAGE})=2
# ANSWER2=EVAL(expression="'yes' if ANSWER0 > 0 or ANSWER1 > 0 else 'no'")=EVAL(expression="'yes' if 0 > 0 or 2 > 0 else 'no'")=yes
# RESULT -> ANSWER2 -> yes
# Explanation:
# There are the red flags {BOX1}.
# """,
# 4
# """Question: The plane above the mountain is higher than what?
# Program:
# BOX0=LOC(image={IMAGE}, object='mountain')={BOX0_IMAGE}
# IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
# BOX1=LOC(image={IMAGE0}, object='plane')={BOX1_IMAGE}
# IMAGE1=CROP_ABOVE(bbox={BOX1_IMAGE})={IMAGE1}
# ANSWER0=VQA(image={IMAGE1}, question='What is the plane higher than?')=another plane
# RESULT -> ANSWER0 -> another plane
# Explanation:
# The plane {BOX1} above the mountain {BOX0} is higher than another plane.
# """,
# 5
# """Question: Who is riding?
# Program:
# BOX0=LOC(image={IMAGE}, object='riding')={BOX0_IMAGE}
# IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
# ANSWER0=VQA(image={IMAGE0}, question='Who is riding?')=snowboarder
# RESULT -> ANSWER0 -> snowboarder
# Explanation:
# The person {BOX0} riding is a snowboarder.
# """,
# 6
"""Question: Is the cup on the left side or on the right?
Program:
BOX0=LOC(image={IMAGE}, object='LEFT')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE0}, object='cup')={BOX1_IMAGE}
ANSWER0=COUNT(bbox={BOX1_IMAGE})=0
ANSWER1=EVAL(expression="'left' if ANSWER0 > 0 else 'right'")=EVAL(expression="'left' if 0 > 0 else 'right'")=right
RESULT -> ANSWER1 -> right
Explanation:
The cup is not on the left side {BOX0}.
""",
# 7
"""Question: Which color do you think the floor has?
Program:
ANSWER0=VQA(image={IMAGE}, question='Which color do you think the floor has?')=gray
RESULT -> ANSWER0 -> gray
Explanation:
The floor is gray.
""",
# 8
"""Question: Are there either glasses or ties in the image?
Program:
BOX0=LOC(image={IMAGE}, object='glasses')={BOX0_IMAGE}
BOX1=LOC(image={IMAGE}, object='ties')={BOX1_IMAGE}
ANSWER0=COUNT(bbox={BOX0_IMAGE})=0
ANSWER1=COUNT(bbox={BOX1_IMAGE})=0
ANSWER2=EVAL(expression="'yes' if ANSWER0 + ANSWER1 > 0 else 'no'")=EVAL(expression="'yes' if 0 + 0 > 0 else 'no'")=no
RESULT -> ANSWER2 -> no
Explanation:
There is neither glasses nor a tie.
""",
# 9
"""Question: Does the woman wear gloves?
Program:
BOX0=LOC(image={IMAGE}, object='woman')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='Does the woman wear gloves?')=yes
RESULT -> ANSWER0 -> yes
Explanation:
The glove that the woman {BOX0} is wearing.
""",
# 10
# """Question: Is the fence post on top of the wall?
# Program:
# BOX0=LOC(image={IMAGE}, object='wall')={BOX0_IMAGE}
# IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
# BOX1=LOC(image={IMAGE0}, object='fence post')={BOX1_IMAGE}
# ANSWER0=COUNT(bbox={BOX1_IMAGE})=0
# ANSWER1=EVAL(expression="'yes' if ANSWER0 > 0 else 'no'")=EVAL(expression="'yes' if 0 > 0 else 'no'")=no
# RESULT -> ANSWER1 -> no
# Explanation:
# There is no fence post on the top of the wall {BOX0}.
# """,
# 11
"""Question: Do you see either a bench or a skateboard in the picture?
Program:
BOX0=LOC(image={IMAGE}, object='bench')={BOX0_IMAGE}
BOX1=LOC(image={IMAGE}, object='skateboard')={BOX1_IMAGE}
ANSWER0=COUNT(bbox={BOX0_IMAGE})=0
ANSWER1=COUNT(bbox={BOX1_IMAGE})=3
ANSWER2=EVAL(expression="'yes' if ANSWER0 + ANSWER1 > 0 else 'no'")=EVAL(expression="'yes' if 0 + 3 > 0 else 'no'")=yes
RESULT -> ANSWER2 -> yes
Explanation:
There are the skateboards {BOX1}.
""",
# 12
"""Question: Who is wearing shorts?
Program:
BOX0=LOC(image={IMAGE}, object='shorts')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='Who is wearing shorts?')=man
RESULT -> ANSWER0 -> man
Explanation:
The person {IMAGE0} wearing the shorts {BOX0} is a man.
""",
# 13
"""Question: Is the color of the table different than the color of the floor?
Program:
BOX0=LOC(image={IMAGE}, object='table')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE}, object='floor')={BOX1_IMAGE}
IMAGE1=CROP(bbox={BOX1_IMAGE})={IMAGE1}
ANSWER0=VQA(image={IMAGE0}, question='What color is the table?')=black
ANSWER1=VQA(image={IMAGE1}, question='What color is the floor?')=gray
ANSWER2=EVAL(expression="'yes' if ANSWER0 != ANSWER1 else 'no'")=EVAL(expression="'yes' if 'black' != 'gray' else 'no'")=yes
RESULT -> ANSWER2 -> yes
Explanation:
The table {BOX0} is black, the floor {BOX1} is gray.
""",
# 14
# """Question: Is the microwave oven made of stainless steel silver and rectangular?
# Program:
# BOX0=LOC(image={IMAGE}, object='microwave oven')={BOX0_IMAGE}
# IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
# ANSWER0=VQA(image={IMAGE0}, question='What material is the microwave oven made of?')=stainless steel
# ANSWER1=VQA(image={IMAGE0}, question='What shape is the microwave oven?')=square
# ANSWER2=EVAL(expression="'yes' if ANSWER0 == 'stainless steel silver' and ANSWER1 == 'rectangular' else 'no'")=EVAL(expression="'yes' if 'stainless steel' == 'stainless steel silver' and 'square' == 'rectangular' else 'no'")=no
# RESULT -> ANSWER2 -> no
# Explanation:
# The stainless steel microwave oven {BOX0} is silver and square.
# """,
# 15
"""Question: Is the chair that is not full white or black?
Program:
BOX0=LOC(image={IMAGE}, object='chair')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='What color is the chair?')=white
ANSWER1=EVAL(expression="'yes' if ANSWER0 != 'white' and ANSWER0 != 'black' else 'no'")=EVAL(expression="'yes' if 'white' != 'white' and 'white' != 'black' else 'no'")=no
RESULT -> ANSWER1 -> no
Explanation:
The chair {BOX0} that is not full is white.
""",
# 16
"""Question: Is the baseball mitt made of leather black and open?
Program:
BOX0=LOC(image={IMAGE}, object='baseball mitt')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='What material is the baseball mitt made of?')=leather
ANSWER1=VQA(image={IMAGE0}, question='What color is the baseball mitt?')=black
ANSWER2=VQA(image={IMAGE0}, question='Is the baseball mitt open?')=yes
ANSWER3=EVAL(expression="'yes' if ANSWER0 == 'leather' and ANSWER1 == 'black' and ANSWER2 == 'yes' else 'no'")=EVAL(expression="'yes' if 'leather' == 'leather' and 'black' == 'black' and yes == 'yes' else 'no'")=yes
RESULT -> ANSWER3 -> yes
Explanation:
The baseball mitt {BOX0} made of leather is black and open.
""",
# 17
"""Question: Is the man that is to the right of the woman wearing a suit?
Program:
BOX0=LOC(image={IMAGE}, object='woman')={BOX0_IMAGE}
IMAGE0=CROP_RIGHTOF(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE0}, object='man')={BOX1_IMAGE}
IMAGE1=CROP(bbox={BOX1_IMAGE})={IMAGE1}
ANSWER0=VQA(image={IMAGE1}, question='What is the man wearing?')=tuxedo
ANSWER1=EVAL(expression="'yes' if ANSWER0 == 'suit' else 'no'")=EVAL(expression="'yes' if 'tuxedo' == 'suit' else 'no'")=no
RESULT -> ANSWER1 -> no
Explanation:
There is no suit that the man {BOX1} to the right of the woman {BOX0} is wearing.
""",
# 18
"""Question: What is the bathtub made of?
Program:
BOX0=LOC(image={IMAGE}, object='bathtub')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='What is the bathtub made of?')=porcelain
RESULT -> ANSWER0 -> porcelain
Explanation:
The bathtub {BOX0} is porcelain.
""",
# 19
# """Question: Are the drawers to the left of the clean oven?
# Program:
# BOX0=LOC(image={IMAGE}, object='oven')={BOX0_IMAGE}
# IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
# BOX1=LOC(image={IMAGE0}, object='drawers')={BOX1_IMAGE}
# IMAGE1=CROP_LEFTOF(bbox={BOX1_IMAGE})={IMAGE1}
# ANSWER0=COUNT(bbox={BOX1_IMAGE})=4
# ANSWER1=EVAL(expression="'yes' if ANSWER0 > 0 else 'no'")=EVAL(expression="'yes' if 4 > 0 else 'no'")=yes
# RESULT -> ANSWER1 -> yes
# Explanation:
# The drawers {BOX1} is to the left of the clean oven {BOX0}.
# """,
# 20
"""Question: Does the man below the frame sit on a couch?
Program:
BOX0=LOC(image={IMAGE}, object='frame')={BOX0_IMAGE}
IMAGE0=CROP_BELOW(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE0}, object='man')={BOX1_IMAGE}
IMAGE1=CROP(bbox={BOX1_IMAGE})={IMAGE1}
BOX2=LOC(image={IMAGE1}, object='couch')={BOX2_IMAGE}
ANSWER0=COUNT(bbox={BOX2_IMAGE})=1
ANSWER1=EVAL(expression="'yes' if ANSWER0 > 0 else 'no'")=EVAL(expression="'yes' if 1 > 0 else 'no'")=yes
RESULT -> ANSWER1 -> yes
Explanation:
The couch {BOX2} that the man {BOX1} below the frame {BOX0} is sitting on.
""",
# 21
"""Question: What animal is to the left of the keyboard?
Program:
BOX0=LOC(image={IMAGE}, object='keyboard')={BOX0_IMAGE}
IMAGE0=CROP_LEFTOF(bbox={BOX0_IMAGE})={IMAGE0}
ANSWER0=VQA(image={IMAGE0}, question='What animal is in the image?')=dog
RESULT -> ANSWER0 -> dog
Explanation:
The animal to the left of the keyboard {BOX0} is a dog.
""",
# 22
"""Question: Are the glass and the picture frame made of the same material?
Program:
BOX0=LOC(image={IMAGE}, object='glass')={BOX0_IMAGE}
IMAGE0=CROP(bbox={BOX0_IMAGE})={IMAGE0}
BOX1=LOC(image={IMAGE}, object='picture frame')={BOX1_IMAGE}
IMAGE1=CROP(bbox={BOX1_IMAGE})={IMAGE1}
ANSWER0=VQA(image={IMAGE0}, question='What material is the glass made of?')=glass
ANSWER1=VQA(image={IMAGE1}, question='What material is the picture frame made of?')=wood
ANSWER2=EVAL(expression="'yes' if ANSWER0 == ANSWER1 else 'no'")=EVAL(expression="'yes' if 'glass' == 'wood' else 'no'")=no
RESULT -> ANSWER2 -> no
Explanation:
The glass {BOX0} is made of glass, the picture frame {BOX1} is made of wood.
"""
]


def create_prompt(inputs, num_prompts=8, method='random', seed=42):
    if method == 'all':
        prompt_examples = GQA_CURATED_EXAMPLES
    elif method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Translate the program steps of a visual question to its explanation straightforward and easy to understand.\n\n{prompt_examples}'

    return prompt_examples + "\nQuestion: {question}\nProgram: {program}\nExplanation:".format(**inputs)
