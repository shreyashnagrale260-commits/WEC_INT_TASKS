from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("lucadiliello/newsqa", split="validation[:100]")  
QA_MODEL = "deepset/roberta-base-squad2"   
TRANS_MODEL = "Helsinki-NLP/opus-mt-en-fr" 

qa_pipe = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL)
trans_pipe = pipeline("translation_en_to_fr", model=TRANS_MODEL, tokenizer=TRANS_MODEL)


def get_french_answer(question, context):
    try:
        qa_out = qa_pipe({"question": question, "context": context})
        ans_en = qa_out["answer"].strip()
        ans_fr = trans_pipe(ans_en)[0]["translation_text"]
        return ans_en, ans_fr
    except Exception as e:
        return "", ""


if __name__ == "__main__":
    results = []
    for item in dataset:
        q = item["question"]
        context = item["context"]
        ans_en, ans_fr = get_french_answer(q, context)
        results.append({"question": q, "answer_en": ans_en, "answer_fr": ans_fr})

    for r in results[:5]:
        print("\nQuestion:", r["question"])
        print("Answer (EN):", r["answer_en"])
        print("Answer (FR):", r["answer_fr"])
