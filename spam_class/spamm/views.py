from django.shortcuts import render
from pickle import load
from pathlib import Path
import nltk
import string

# Get the directory of this file
BASE_DIR = Path(__file__).resolve().parent

# Correct paths to the model and vectorizer
tfidf = load(open(BASE_DIR / 'vectorizer.pkl', 'rb'))
model = load(open(BASE_DIR / 'model.pkl', 'rb'))
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(nltk.stem.PorterStemmer().stem(i))
        
    return " ".join(y) 

def form(request):
    if request.method == 'POST':
        message = request.POST['message']
        data = [transform_text(message)]
        vect = tfidf.transform(data).toarray()
        my_prediction = model.predict(vect)
        if my_prediction[0] == 1:
            my_prediction = 'Spam'
        else:
            my_prediction = 'Not Spam'
        return render(request, 'index.html', {'prediction': my_prediction})
    return render(request, 'index.html')
