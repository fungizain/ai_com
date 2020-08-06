import socketio

from config import endpoint, token, make_directory
from model import AI_Model

sio = socketio.Client()
sio.connect("http://%s?token=%s" % (endpoint, token), namespaces=["/contest"])

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


    if question_tc in ("S001", "S003"):
        answer = {
            "choiceID": choices[ans_ind]
        }
    elif question_tc == "S004":

    elif question_tc in ("S002", "M001"):

    elif question_tc == "D001":


    sio.emit("answer", {
        "questionId": question_id,
        "answer": answer
    }, namespace="/contest")

    # save question string
    make_directory('question/')
    now = round(time.time())
    with open(f"question/q_{now}.txt", 'w') as f:
        f.write(str(question))