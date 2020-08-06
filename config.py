import os
import io
import ast
import time
import glob
from base64 import b64decode
from PIL import Image
import numpy as np

endpoint = "api01.aiplug.local:5000"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoicG9wYXJlZCIsInVzZXJfdHlwZSI6IkNPTlRFU1RBTlQiLCJpYXQiOjE1OTY2MTQzNzQsImV4cCI6MTYwMTc5ODM3NH0.qtb6qcq0juJ97EQIFZh7r_DbQTqTQ0VDfRSTCqlXSOI"


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# for saved data {{{
def string_save_image(img_str, save_path):
    img_str += "==="
    img_str = b64decode(img_str)
    with open(save_path, 'wb') as f:
        f.write(img_str)

def saved_question_parse_(data):
    if isinstance(data, str):
        data = ast.literal_eval(data)
    assert isinstance(data, dict)

    question_id = data["id"]
    question_t = data["type"]
    question_tc = data["typeCode"]

    folder = os.path.join('question/', f'{question_t}_{question_id}/')
    content = data["content"]

    for k, v in content.items():
        if isinstance(v, str):
            name = os.path.join(folder, f'{k}.jpg')
            string_save_image(v, name)
        elif isinstance(v, (list, tuple)):
            for _ in v:
                name = os.path.join(folder, f'{_["id"]}.jpg')
                string_save_image(_["imageString"], name)

def saved_question_parse():
    q_list = glob.glob('question/q_*.txt')
    for q in q_list:
        print(f"Parsing: {q}")
        with open(q, 'r') as f:
            data = f.read()
        saved_question_parse_(data)
# }}}


# for answering {{{
def string_to_image(img_str, to_nparray=False):
    img_str += "==="
    img_str = b64decode(img_str)

    img = Image.open(io.BytesIO(img_str))
    if not to_nparray:
        return np.uint8(img)
    return img

def parse_question(content, question_tc):

    if question_tc in ("S001", "S003"):
        in_1 = decode_to_image(content["subjectImageString"])
        in_2 = []
        choices = []
        for choice in content["choices"]:
            in_2.append(decode_to_image(choice["imageString"]))
            choices.append(choice["id"])
    elif question_tc == "S004":
        in_1 = decode_to_image(content["subjectImageString"])
        in_2 = []
        choices = ["A", "B"] # actor, actress
    elif question_tc in ("S002", "M001"):
        in_1 = decode_to_image(content["groupImageString"])
        in_2 = []
        choices = []
        for choice in content["choices"]:
            in_2.append(decode_to_image(choice["imageString"]))
            choices.append(choice["id"])
    elif question_tc == "D001":
        in_1 = decode_to_image(content["subjectImageString"])
        in_2 = decode_to_image(content["groupImageString"])
        choices = []
    else:
        print("Can't match the question type code!")
        return False

    return in_1, in_2, choices
# }}}