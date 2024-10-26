import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data():
    news_df = pd.read_csv('data/news_articles.csv')
    stock_df = pd.read_csv('data/stock_prices.csv')
    return news_df, stock_df

def preprocess_data(news_df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    news_df['tokens'] = news_df['article'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length'))
    return news_df
