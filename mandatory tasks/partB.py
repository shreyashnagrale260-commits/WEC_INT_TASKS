from transformers import MarianMTModel, MarianTokenizer


model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


english_sentences = ["Hi, I am Shreyash" , "I am preparing for web Enthusiasts club", "I like to be part of intelligence sig in it"]


tokens = tokenizer(english_sentences, return_tensors="pt", padding=True)
translated_tokens = model.generate(**tokens)


spanish_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]


for es in spanish_sentences:
    print(es)
