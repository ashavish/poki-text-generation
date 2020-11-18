# Generate Fast Text Embeddings


from gensim.models.fasttext import FastText
import re
import nltk
from gensim.test.utils import get_tmpfile
from gensim.utils import tokenize
from gensim import utils
from gensim.test.utils import datapath


data_path = "./poki.txt"

class MyIter:
    def __iter__(self):        
        with utils.open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))


ft_model = FastText(size=60, window=20, min_count=1)
ft_model.build_vocab(sentences=MyIter())
total_examples = ft_model.corpus_count
ft_model.train(sentences=MyIter(), total_examples=total_examples, epochs=20)

ft_model.wv.most_similar(['red'], topn=5)


fname = get_tmpfile("fasttext_poki.model")
ft_model.save(fname)
model = FastText.load(fname)

