from flask import Flask,render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
import tempfile
from werkzeug.utils import secure_filename
import gensim, sklearn

data = pd.read_pickle('env\data.pkl')
#D:\data science workshop - datai2i\cluster champs\env\data.pkl
#data = pickle.load(open('env\data.pkl','rb'))
with open('env\model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('env\\topic_mapping.pkl', 'rb') as f:
    topic_mapping = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           data = data )

@app.route('/upload', methods=['POST'])
def upload():
    
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_filename = tmp.name
 
    topic_df = process_pdf(tmp_filename)
    os.remove(tmp_filename)
    topic_cluster = str(topic_df['topic_name'][0])
    return redirect(url_for('result', cluster=topic_cluster))

@app.route('/result', methods=['GET'])
def result():
    cluster = request.args.get('cluster')
    return render_template('result.html', cluster=cluster, data = data)

def process_pdf(pdf_path):
    from PyPDF2 import PdfReader
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from gensim import corpora
    from dataprep.clean import clean_text
    def extract_text_from_pdf(pdf_path):
        df2 = pd.DataFrame(columns=['PDF', 'Text'])
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page_num in range(25):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        pdf = pdf_path.split('/')[-1]
        new_row = pd.Series({'PDF': pdf, 'Text': text})
        df2 = pd.concat([df2, new_row.to_frame().T], ignore_index=True)
        df2=clean_text(df2,"Text")
        def tokenize_text(text):
            if isinstance(text, str):
                return word_tokenize(text)
            else:
                return []
        df2['cleaned_text'] = df2['Text'].apply(tokenize_text)
        stop_words = stopwords.words('english')
        common_words = ['use','one','chapter','book','use ',' use', ' use ','coil','coil ', ' coil',' coil ','ooo','using','used']
        stop_list = stop_words+common_words
        def clean_tokens(tokens):
            filtered_tokens = [token for token in tokens if token not in stop_list]
            filtered_tokens = [token for token in tokens if len(token) > 2]
            return filtered_tokens
        df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x: clean_tokens(x))
        return df2
    path = pdf_path
    data = extract_text_from_pdf(path)
    def build_corpus(column):
        df2 = data.copy()
        processed_text = [document for document in df2[column]]
        dictionary = corpora.Dictionary(processed_text)
        corpus1 = [dictionary.doc2bow(text) for text in processed_text]
        return corpus1
    corpus1 = build_corpus('cleaned_text')
    data['topic'] = list(lda_model[corpus1])

    def get_topic(column):
        df2 = data.copy()
        topic_list = df2[column]
        for index, row in df2.iterrows():
            probabilities = [prob for _, prob in topic_list[index]]
            max_index = probabilities.index(max(probabilities))
            max_topic = topic_list[index][max_index]
            df2.at[index, column] = max_topic
        df2['topic_name'] = ''
        for index, row in df2.iterrows():
            topic_id = row[column][0]
            if topic_id in topic_mapping:
                df2.at[index, 'topic_name'] = topic_mapping[topic_id]

        return df2
    data = get_topic('topic')
    return data

@app.route('/pdfs', methods=['GET'])
def display_pdfs():
    keyword = request.args.get('keyword')
    filtered_pdfs = data[['PDF','url', keyword]][data[keyword] > 0].sort_values(by=keyword, ascending=False)
    return render_template('pdfs.html', keyword=keyword, pdfs=filtered_pdfs, data=data)


if __name__ == '__main__':
    app.run(  debug = True) 
