import TextMining as tm
from joblib import load
import openewfile  as of
import pickle


def loadmodel(filename):
    model = load(filename)
    return model

def preprocess(text, lemmit = True):
    if lemmit:
        text = tm.cleanText(text,fix=SlangS, pattern2 = False, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 3)
        #text = tm.handlingnegation(text)
    else:
        text = tm.cleanText(text,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 3)
    return(text)

def loadtokenizer(filepath):
    with open(filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return (tokenizer)

fSlang = of.openfile(path = './slangword')
bahasa = 'id'
stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = True)
sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
tokenizersen = loadtokenizer(of.openfile(path = './tokenizer_sentiment'))
model = loadmodel(of.openfile(path = './model_sentiment'))

def hoaxpredict(text, lemma = True):
    text = str(text)
    text = [preprocess(text, lemmit = lemmatizer)]
    text = tokenizersen.transform(text)
    hoax = model.predict_proba(text)
    polarityvalid = hoax[0][1]
    return(polarityvalid) 