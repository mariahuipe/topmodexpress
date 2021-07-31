from flask import Flask, render_template, flash, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from gensim import matutils

ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/sample')
def sample():
    return render_template('samplefiles.html')    
        
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      numTopics = 5
      text_loaded = []
      file = request.files.get('file')
      text1=file.read()
      text_loaded=text1.splitlines()
      #print(text1)
      vectorizer = TfidfVectorizer(max_df=0.5, max_features=50000,min_df=2, stop_words='english',use_idf=True)
      X = vectorizer.fit_transform(text_loaded)
      id2words={}
      for i, word in enumerate(vectorizer.get_feature_names()):
          id2words[i] = word
      corpus = matutils.Sparse2Corpus(X, documents_columns=False)
      lda = models.ldamodel.LdaModel(corpus, num_topics=numTopics, id2word=id2words)
      output_text=[]
      for i, item in enumerate(lda.show_topics(num_topics=numTopics, num_words=15, formatted=False)):
          #output_text.append("Topic: " + str(i))
          topic_list=[]
          for (term, weight) in item[1]:
              term_obj = {}
              term_obj[term]= str(round(weight*100,2)) + "%"
              topic_list.append(term_obj)
          output_text.append(topic_list)
      #output_text1=str(output_text)
      return  render_template('display.html',topics=output_text)
           
           