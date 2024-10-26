from transformers import BertForSequenceClassification
import torch

def fine_tune_model(news_df):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # Fine-tuning logic here
    return model

def get_sentiment(model, tokens):
    with torch.no_grad():
        outputs = model(tokens)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions
