import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

train_df = pd.read_csv("../data/train.tsv", sep="\t", encoding = "utf-8", header=None)
test_df = pd.read_csv("../data/test.tsv", sep="\t", encoding = "utf-8", header=None)
dev_df = pd.read_csv("../data/dev.tsv", sep="\t", encoding = "utf-8", header=None)

train_df.columns = ['text', 'emotions', 'id']
test_df.columns = ['text', 'emotions', 'id']
dev_df.columns = ['text', 'emotions', 'id']

train_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), train_df['emotions']))
test_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), test_df['emotions']))
dev_df['emotions'] = list(map(lambda s : list(map(int, s.split(','))), dev_df['emotions']))

def emotions_to_categorical(df):
    ems = """
        admiration
        amusement
        anger
        annoyance
        approval
        caring
        confusion
        curiosity
        desire
        disappointment
        disapproval
        disgust
        embarrassment
        excitement
        fear
        gratitude
        grief
        joy
        love
        nervousness
        optimism
        pride
        realization
        relief
        remorse
        sadness
        surprise
        neutral
    """
    res = []

    for i in df['emotions']:
        tmp = [0 for _ in range(28)]
        for j in i:
            tmp[j] = 1
        res.append(tmp)
    tmp_df = pd.DataFrame(res, columns=ems.split())
    
    return tmp_df

train_df = pd.concat([train_df, emotions_to_categorical(train_df)], axis=1)
test_df = pd.concat([test_df, emotions_to_categorical(test_df)], axis=1)
dev_df = pd.concat([dev_df, emotions_to_categorical(dev_df)], axis=1)

train_df = train_df.drop(columns=['emotions', 'id'])
test_df = test_df.drop(columns=['emotions', 'id'])
dev_df = dev_df.drop(columns=['emotions', 'id'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_labels = 28
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

ems = """
        admiration
        amusement
        anger
        annoyance
        approval
        caring
        confusion
        curiosity
        desire
        disappointment
        disapproval
        disgust
        embarrassment
        excitement
        fear
        gratitude
        grief
        joy
        love
        nervousness
        optimism
        pride
        realization
        relief
        remorse
        sadness
        surprise
        neutral
    """
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        labels = self.data.loc[index, ems.split()].values
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

# optimizer 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 학습 시작
model.train()
epoch_num = 3
for epoch in range(epoch_num):
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epoch_num}')
    total_loss = 0
    
    for batch in progress_bar:
        # DataLoader가 반환하는 딕셔너리의 키를 올바르게 접근
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 라벨 컬럼을 'labels' 키로 묶어 전달
        # 데이터셋 클래스에서 이 부분 처리를 해야 함 (이전 답변 참조)
        labels = batch['labels'].to(device)

        # model의 출력
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 손실 계산 및 역전파
        # AutoModelForSequenceClassification은 'problem_type'이 'multi_label_classification'이면
        # 자동으로 Binary Cross-Entropy Loss를 계산하여 outputs.loss에 담습니다.
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Progress bar 업데이트
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epoch_num} completed! Average Loss: {avg_loss:.4f}")

# 저장할 디렉토리 경로 지정
save_directory = "./model_saved_bert_base_v2"

# 모델과 토크나이저 저장
model.save_pretrained(save_directory, from_pt=True)
tokenizer.save_pretrained(save_directory, from_pt=True)
