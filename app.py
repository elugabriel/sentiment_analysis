from flask import Flask, render_template, request, redirect, url_for
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Import the stopwords from NLTK
stop_words = set(stopwords.words('english'))

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        transactions = request.form['text']

        def data_processing(text):
            text = text.lower()
            text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'\@\w+|\#', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text_tokens = word_tokenize(text)
            filtered_text = [w for w in text_tokens if not w in stop_words]
            return " ".join(filtered_text)

        transactions = data_processing(transactions)  # Call the data_processing function

        # Sentiment Analysis using VADER
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(transactions)

        # Determine the sentiment label based on the compound score
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return render_template('result.html', sentiment=sentiment)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
