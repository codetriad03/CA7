import requests

# News API credentials
API_KEY = '31ba7958f4bc4bde9fb76433095711a3'
url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'

response = requests.get(url)
data = response.json()

with open('news_headlines.txt', 'a') as f:
    for article in data['articles']:
        f.write(article['title'] + '\n')
