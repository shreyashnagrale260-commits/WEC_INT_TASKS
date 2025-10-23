from transformers import pipeline
from datasets import load_dataset
import evaluate

dataset = load_dataset("newsqa", split="validation")
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
em = evaluate.load("exact_match")
f1 = evaluate.load("f1")

ems, f1s = [], []
for d in dataset:
    if not d["answer"]["text"]: continue
    pred = qa(question=d["question"], context=d["story_text"])["answer"]
    ems.append(em.compute(predictions=[pred], references=[d["answer"]["text"][0]])["exact_match"])
    f1s.append(f1.compute(predictions=[pred], references=[d["answer"]["text"][0]])["f1"])

print("Average EM:", sum(ems)/len(ems))
print("Average F1:", sum(f1s)/len(f1s))
