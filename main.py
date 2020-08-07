import time
import socketio

from config import endpoint, token, make_directory, parse_question
from model import AI_Model

sio = socketio.Client()
sio.connect("http://%s?token=%s" % (endpoint, token), namespaces=["/contest"])

model = AI_Model()

@sio.on("health-check", namespace="/contest")
def on_health_check():
    return {"ok": True}

@sio.on("question", namespace="/contest")
def on_question(question):

    question_id = question["id"]
    question_t = question["type"]
    question_tc = question["typeCode"]
    content = question["content"]

    in_1, in_2, choices = parse_question(content, question_tc)
    ans_ind = model.compare(in_1, in_2, question_tc)

    if question_tc == "M001":
        answer = {
            "choiceID": [choices[_] for _ in ans_ind]
        }
    elif question_tc == "D001":
        {
            "points": [
                {"x": ans_ind[0][0], "y": ans_ind[0][1]},
                {"x": ans_ind[1][0], "y": ans_ind[1][1]}
            ]
        }
    else:
        answer = {
            "choiceID": choices[ans_ind]
        }

    sio.emit("answer", {
        "questionId": question_id,
        "answer": answer
    }, namespace="/contest")

    # save question string
    make_directory('question/')
    now = round(time.time())
    with open(f"question/q_{now}.txt", 'w') as f:
        f.write(str(question))