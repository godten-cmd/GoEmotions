import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from wordcloud import WordCloud
from collections import Counter
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

train_df = pd.read_csv("../data/train.tsv", sep="\t", encoding = "utf-8", header=None)
test_df = pd.read_csv("../data/test.tsv", sep="\t", encoding = "utf-8", header=None)
dev_df = pd.read_csv("../data/dev.tsv", sep="\t", encoding = "utf-8", header=None)

train_df.columns = ['text', 'emotions', 'id']
test_df.columns = ['text', 'emotions', 'id']
dev_df.columns = ['text', 'emotions', 'id']

train_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), train_df['emotions']))
test_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), test_df['emotions']))
dev_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), dev_df['emotions']))

def emotions_to_ekman(df):
    # anger disgust fear joy sadness surprise neutral
    ekman = [3, 3, 0, 0, 3, 3, 5, 5, 3, 4, 0, 1, 4, 3, 2, 3, 4, 3, 3, 2, 3, 3, 5, 3, 4, 4, 5, 6]
    res = []

    for i in df['emotions']:
        tmp = [0, 0, 0, 0, 0, 0, 0]
        for j in i:
            tmp[ekman[j]] = 1
        res.append(tmp)
    tmp_df = pd.DataFrame(res, columns=['angry', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'])
    
    return tmp_df

train_df = pd.concat([train_df, emotions_to_ekman(train_df)], axis=1)
test_df = pd.concat([test_df, emotions_to_ekman(test_df)], axis=1)
dev_df = pd.concat([dev_df, emotions_to_ekman(dev_df)], axis=1)

train_df = train_df.drop(columns=['emotions', 'id'])
test_df = test_df.drop(columns=['emotions', 'id'])
dev_df = dev_df.drop(columns=['emotions', 'id'])

# 학습된 모델 및 토크나이저 불러오기
load_directory = "./model_saved_bert_base"

model = AutoModelForSequenceClassification.from_pretrained(load_directory)
tokenizer = AutoTokenizer.from_pretrained(load_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        labels = self.data.loc[index, ['angry', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']].values
        labels = labels.astype('float32')
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float)
        
        return item


MAX_LEN = 512
BATCH_SIZE = 32

train_dataset = EmotionDataset(train_df, tokenizer, MAX_LEN)
test_dataset = EmotionDataset(test_df, tokenizer, MAX_LEN)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for batch in train_loader:
    print(batch)
    break

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve

def evaluate_model(model, test_loader, device, emotion_names = ['angry', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']):
    """
    다중 라벨 분류 모델을 평가하는 통합 함수.
    정확도, F1-점수를 계산하고, 각 라벨에 대한 개별 Precision-Recall 곡선을 그립니다.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    accuracy = accuracy_score(all_labels, all_predictions)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='micro'
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    precision_macro_std = np.std(precision_per_label)
    recall_macro_std = np.std(recall_per_label)
    f1_macro_std = np.std(f1_per_label)

    print("--- 모델 평가 결과 ---")
    print(f"전체 샘플에 대한 정확도 (Exact Match Accuracy): {accuracy:.4f}")
    print("\n--- Micro 평균 지표 ---")
    print(f"Precision (Micro): {precision_micro:.4f}")
    print(f"Recall (Micro): {recall_micro:.4f}")
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    print("\n--- Macro 평균 지표 ---")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    print("\n--- 라벨별 지표 ---")
    for i in range(len(emotion_names)):
        print(f"{emotion_names[i]} - Precision: {precision_per_label[i]:.4f}, Recall: {recall_per_label[i]:.4f}, F1-Score: {f1_per_label[i]:.4f}")
    
    print(f"\nPrecision (Macro) 표준편차: {precision_macro_std:.4f}")
    print(f"Recall (Macro) 표준편차: {recall_macro_std:.4f}")
    print(f"F1-Score (Macro) 표준편차: {f1_macro_std:.4f}")
    
    # 각 감정별 Precision-Recall 곡선 그리기
    n_classes = all_labels.shape[1]
    for i in range(n_classes):
        plt.figure(figsize=(6, 5))
        
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        
        plt.plot(recall_curve, precision_curve, label=f'{emotion_names[i]}')
        plt.plot([0, 1], [1, 0], linestyle='--', color='gray')
        plt.xlabel('재현율 (Recall)')
        plt.ylabel('정밀도 (Precision)')
        plt.title(f'정밀도-재현율 곡선: {emotion_names[i]}')
        plt.grid(True)
        plt.legend()
        plt.show()

    return accuracy, f1_micro, f1_macro

accuracy, f1_micro, f1_macro = evaluate_model(model, test_loader, device)


"""
--- 모델 평가 결과 ---
전체 샘플에 대한 정확도 (Exact Match Accuracy): 0.6068

--- Micro 평균 지표 ---
Precision (Micro): 0.7111
Recall (Micro): 0.6568
F1-Score (Micro): 0.6828

--- Macro 평균 지표 ---
Precision (Macro): 0.6585
Recall (Macro): 0.5910
F1-Score (Macro): 0.6217

--- 라벨별 지표 ---
angry - Precision: 0.5922, Recall: 0.4780, F1-Score: 0.5290
disgust - Precision: 0.5619, Recall: 0.4797, F1-Score: 0.5175
fear - Precision: 0.7215, Recall: 0.5816, F1-Score: 0.6441
joy - Precision: 0.8489, Recall: 0.7928, F1-Score: 0.8199
sadness - Precision: 0.6324, Recall: 0.5673, F1-Score: 0.5981
surprise - Precision: 0.5819, Recall: 0.6189, F1-Score: 0.5999
neutral - Precision: 0.6707, Recall: 0.6189, F1-Score: 0.6438

Precision (Macro) 표준편차: 0.0931
Recall (Macro) 표준편차: 0.0986
F1-Score (Macro) 표준편차: 0.0932
"""