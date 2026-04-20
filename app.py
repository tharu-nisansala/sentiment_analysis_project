from flask import Flask, render_template, request,redirect
from helper import preprocess_text, vectorizer, get_prediction, tokens
app = Flask(__name__)

reviews = []
positive = 0
negative = 0

@app.route('/')
def index():

    data = {}
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data=data)

@app.route('/', methods=['POST'])
def my_post():
    text = request.form['review']
    preprocessed_txt = preprocess_text(text)
    vectorized_txt = vectorizer([preprocessed_txt], tokens)
    prediction = get_prediction(vectorized_txt)
    if prediction == 'positive':
        global positive
        positive += 1
    else:
        global negative
        negative += 1
    reviews.insert(0, text)
    return redirect(request.url)

if __name__ == '__main__':
    app.run()