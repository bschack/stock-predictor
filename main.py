import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import tensorflow as tf

# Download the VADER lexicon
nltk.download('vader_lexicon')


def get_articles(ticker):
    search_url = f"https://finance.yahoo.com/quote/{ticker}/news/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    search_div = soup.find('div', class_='news-stream')
    articles = []

    if search_div:
        search_soup = BeautifulSoup(str(search_div), 'html.parser')
        for item in search_soup.find_all('li')[:10]:
            link = item.find('a')
            if link:
                articles.append(link['href'])

    return articles


def scrape_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract article content based on typical paragraph tags
    paragraphs = soup.find_all('p')
    article_text = ' '.join(p.get_text() for p in paragraphs)

    return article_text


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


def get_gainers_and_losers():
    url_gainers = "https://finance.yahoo.com/gainers"
    url_losers = "https://finance.yahoo.com/losers"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    gainers_page = requests.get(url_gainers, headers=headers)
    losers_page = requests.get(url_losers, headers=headers)

    def parse_table(soup):
        table = soup.find("table")
        rows = table.find_all("tr")
        data = {}
        for row in rows[1:]:
            cols = row.find_all("td")
            # data.append({
            #     "symbol": cols[0].text.strip(),
            #     "name": cols[1].text.strip(),
            #     "price": cols[2].text.strip(),
            #     "change": cols[3].text.strip(),
            #     "percent_change": cols[4].text.strip()
            # })
            # add symbol and percent change to data
            data[cols[0].text.strip()] = float(cols[4].text.strip()[:-1])
        return data

    gainers_soup = BeautifulSoup(gainers_page.content, "html.parser")
    gainers = parse_table(gainers_soup)

    losers_soup = BeautifulSoup(losers_page.content, "html.parser")
    losers = parse_table(losers_soup)

    data_size = 10
    negative_bias = 2

    losers = dict(list(losers.items())[-data_size+negative_bias:])
    gainers = dict(list(gainers.items())[-data_size:])

    print(f"Gainers {len(gainers)}/{len(losers)} Losers")

    # concat dictionaries to return a single dictionary
    data = {**gainers, **losers}

    return data


def main(company_name):
    articles = get_articles(company_name)
    sentiments = []

    for i, article in enumerate(articles):
        print(
            f"\r({company_name}): Analyzing article {i+1}/{len(articles)}", end="")
        try:
            article_content = scrape_article_content(article)
            sentiment = analyze_sentiment(article_content)
            sentiments.append(sentiment)
            # print(f"Sentiment: {sentiment}")
        except Exception as e:
            print(f"Failed to scrape or analyze article: {e}")
    print()

    # Calculate average sentiment
    avg_sentiment = {
        "neg": sum(s['neg'] for s in sentiments) / len(sentiments),
        "neu": sum(s['neu'] for s in sentiments) / len(sentiments),
        "pos": sum(s['pos'] for s in sentiments) / len(sentiments),
        "compound": sum(s['compound'] for s in sentiments) / len(sentiments)
    }
    change = -0.0016
    return avg_sentiment, change


def data_prep(sent, change):
    # Example sentiment data (most recent data)
    sentiment_data = pd.DataFrame(sent)

    # Example stock data (most recent stock changes)
    stock_data = pd.DataFrame({
        'Pct Change': change
    })

    # Combine sentiment and stock data
    data = pd.concat([sentiment_data, stock_data], axis=1)

    # Prepare features and target
    X = data[['neg', 'neu', 'pos', 'compound']]
    y = data['Pct Change']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train(X_train_scaled, X_test_scaled, y_train, y_test):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
        64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    # Linear activation for regression
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=50,
                        batch_size=10, validation_split=0.1, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Mean Squared Error: {loss}')
    return model


def predict(model, scaler, ticker):
    sent = main(ticker)

    latest_sentiment_data = pd.DataFrame({
        'neg': [sent[0]['neg']],
        'neu': [sent[0]['neu']],
        'pos': [sent[0]['pos']],
        'compound': [sent[0]['compound']]
    })

    # Preprocess latest data
    latest_sentiment_scaled = scaler.transform(latest_sentiment_data)

    # Predict stock change
    predicted_change = model.predict(latest_sentiment_scaled)
    print(f'Predicted Stock Change: {predicted_change[0][0]}')


if __name__ == "__main__":
    mode = 'train'

    if mode == 'train':
        sent = []
        train_data = get_gainers_and_losers()
        for n in train_data.keys():
            res = main(n)
            sent.append(res[0])
            # change.append(res[1])

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = data_prep(
            sent, train_data.values())
        model = train(X_train_scaled, X_test_scaled, y_train, y_test)

        # save model
        model.save('model.h5')
        joblib.dump(scaler, "scaler.save")
        print("Model saved")

    elif mode == 'predict':
        # load model
        model = tf.keras.models.load_model('model.h5')
        scaler = joblib.load("scaler.save")

        predict(model, scaler, 'SPOT')

    elif mode == 'test':
        print(get_gainers_and_losers())
