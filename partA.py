from datasets import load_dataset
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

nltk.download('punkt')

ds = load_dataset("lucadiliello/newsqa", split="train[:2%]")  # small portion for testing

texts = [item["context"] for item in ds if item["context"]]

sentences = [word_tokenize(txt.lower()) for txt in texts]


cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=0)
pd.DataFrame([(w, cbow.wv[w].tolist()) for w in cbow.wv.index_to_key],
             columns=["word","embedding"]).to_csv("embeddings_cbow.csv", index=False)

skipgram = Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=1)
pd.DataFrame([(w, skipgram.wv[w].tolist()) for w in skipgram.wv.index_to_key],
             columns=["word","embedding"]).to_csv("embeddings_skipgram.csv", index=False)

fasttext = FastText(sentences, vector_size=100, window=5, min_count=3)
pd.DataFrame([(w, fasttext.wv[w].tolist()) for w in fasttext.wv.index_to_key],
             columns=["word","embedding"]).to_csv("embeddings_fasttext.csv", index=False)

print("All three embeddings saved: CBOW, Skip-Gram, FastText")
