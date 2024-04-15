import random

GQA_CURATED_EXAMPLES=[
# 1
"""Question: On which side of the photo is the old vehicle?
Answer:
left
Explanation:
The old vehicle {BOX0} is located here {BOX1}.
Boxes:
{BOX0}: [0, 102, 175, 303]
{BOX1}: [0, 102, 175, 303]
""",
# 2
"""Question: Is the license plate both black and small?
Answer:
yes
Explanation:
There is a black and small license plate {BOX0}.
Boxes:
{BOX0}: [118, 382, 294, 474]
""",
# 6
"""Question: Is the cup on the left side or on the right?
Answer:
right
Explanation:
The cup {BOX0} is on the left.
Boxes:
{BOX0}: [14, 181, 123, 349]
""",
# 7
"""Question: Which color do you think the floor has?
Answer:
gray
Explanation:
The floor {BOX0} is gray.
Boxes:
{BOX0}: [56, 275, 432, 376]
""",
# 8
"""Question: Are there either glasses or ties in the image?
Answer:
no
Explanation:
There is neither glasses nor a tie.
""",
# 9
"""Question: Does the woman wear gloves?
Answer:
no
Explanation:
There are no gloves on the woman {BOX0}.
Boxes:
{BOX0}: [414, 106, 498, 332]
""",
# 11
"""Question: Do you see either a bench or a skateboard in the picture?
Answer:
yes
Explanation:
There are the skateboards {BOX0}.
Boxes:
{BOX0}: [731, 445, 859, 540]
""",
# 12
"""Question: Who is wearing shorts?
Answer:
man
Explanation:
The person {BOX0} wearing the shorts {BOX1} is a man.
Boxes:
{BOX0}: [159, 146, 213, 297]
{BOX1}: [169, 213, 202, 251]
""",
# 13
"""Question: Is the color of the table different than the color of the floor?
Answer:
yes
Explanation:
The table {BOX0} is black, the floor {BOX1} is gray.
Boxes:
{BOX0}: [154, 163, 372, 482]
{BOX1}: [173, 112, 373, 208]
""",
# 15
"""Question: Is the chair that is not full white or black?
Answer:
no
Explanation:
The chair {BOX0} that is not full is white.
Boxes:
{BOX0}: [131, 168, 154, 201]
""",
# 16
"""Question: Is the baseball mitt made of leather black and open?
Answer:
yes
Explanation:
The baseball mitt {BOX0} made of leather is black and open.
Boxes:
{BOX0}: [329, 121, 374, 181]
""",
# 17
"""Question: Is the man that is to the right of the woman wearing a suit?
Answer:
no
Explanation:
There is no suit that the man {BOX1} to the right of the woman {BOX0} is wearing.
Boxes:
{BOX0}: [31, 36, 210, 360]
{BOX1}: [0, 43, 49, 110]
""",
# 18
"""Question: What is the bathtub made of?
Answer:
porcelain
Explanation:
The bathtub {BOX0} is porcelain.
Boxes:
{BOX0}: [73, 305, 244, 375]
""",
# 20
"""Question: Does the man below the frame sit on a couch?
Answer:
yes
Explanation:
The couch {BOX2} that the man {BOX1} below the frame {BOX0} is sitting on.
Boxes:
{BOX0}: [49, 257, 155, 357]
{BOX1}: [196, 201, 271, 307]
{BOX2}: [69, 170, 93, 192]
""",
# 21
"""Question: What animal is to the left of the keyboard?
Answer:
dog
Explanation:
The animal {BOX0} to the left of the keyboard {BOX1} is a dog.
Boxes:
{BOX0}: [14, 135, 185, 327]
{BOX1}: [277, 121, 479, 202]
""",
# 22
"""Question: Are the glass and the picture frame made of the same material?
Answer:
no
Explanation:
The glass {BOX0} is made of glass, the picture frame {BOX1} is made of wood.
Boxes:
{BOX0}: [47, 0, 373, 484]
{BOX1}: [119, 275, 167, 320]
"""
]


def create_prompt(question, num_prompts=8, method='all', seed=42):
    if method == 'all':
        prompt_examples = GQA_CURATED_EXAMPLES
    elif method == 'random':
        random.seed(seed)
        prompt_examples = random.sample(GQA_CURATED_EXAMPLES,num_prompts)
    else:
        raise NotImplementedError

    prompt_examples = '\n'.join(prompt_examples)
    prompt_examples = f'Answer the visual question and generate an explanation with grounded boxes of the objectives in the image. The boxes are in the form of [x1, y1, x2, y2].\n\n{prompt_examples}'

    return prompt_examples + f"\nQuestion: {question}\nAnswer:\n"
