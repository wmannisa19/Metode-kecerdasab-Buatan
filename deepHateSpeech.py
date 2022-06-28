from keras.preprocessing.sequence import pad_sequences
import TextMining as tm
from keras.models import load_model
import tensorflow as tf
import pickle
import openewfile  as of
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def loadtokenizer(filepath):
    with open(filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return (tokenizer)

def loadmodel(filepath):
    global model
    model = load_model(filepath)
    global graph
    graph = tf.get_default_graph()
    return(model)

def preprocess(text, lemmit = True):
    if lemmit:
        text = tm.cleanText(text,fix=SlangS, pattern2 = True, lang = bahasa, lemma=lemmatizer, stops = stops, symbols_remove = True, numbers_remove = True, min_charLen = 3)
        text = tm.handlingnegation(text)
    else:
        text = tm.cleanText(text,fix = SlangS, lang = bahasa, stops = stops, lemma= None,symbols_remove=True,min_charLen=3)
        text = tm.handlingnegation(text)
    return(text)

def predictsentiment(text, lemma = True):
    text = str(text)
    text = [preprocess(text, lemmit = lemma)]
    text = tokenizersen.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=150 ,dtype = 'int32', value = 0)
    with graph.as_default():
        sentiment = model.predict(text,batch_size=1,verbose = 2)[0]
    neg_sc, pos_sc = (-(sentiment[0])), sentiment[1]
    scoresen = neg_sc + pos_sc
    sentimentvalue = []
    if scoresen > 0.1:
        sentimentvalue.append('Positive')
    elif scoresen < -0.1:
        sentimentvalue.append('Negative')
    else:
        sentimentvalue.append('Netral')
        #print(sentiment)
    #sentiment = model._make_predict_function(text)[0]
    #return(getresult(sentiment))
    return(''.join(sentimentvalue))

fSlang = of.openfile(path = './slangword')
bahasa = 'id'
stops, lemmatizer = tm.LoadStopWords(bahasa, sentiment = True)
sw=open(fSlang,encoding='utf-8', errors ='ignore', mode='r');SlangS=sw.readlines();sw.close()
SlangS = {slang.strip().split(':')[0]:slang.strip().split(':')[1] for slang in SlangS}
tokenizersen = loadtokenizer(of.openfile(path = './tokenizer_sentiment'))
model = loadmodel(of.openfile(path = './model_sentiment'))