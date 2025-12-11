import pandas as pd
import numpy as np

print("Veri seti yükleniyor ve bölünüyor...")


df = pd.read_csv('tweets.csv', encoding='latin-1', header=None)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
live_df = df.iloc[split_index:]


train_df.to_csv('tweets_train.csv', index=False, header=False, encoding='latin-1')
live_df.to_csv('tweets_live.csv', index=False, header=False, encoding='latin-1')

print(f"✅ İŞLEM TAMAM!")
print(f"Eğitim Dosyası (tweets_train.csv): {len(train_df)} satır. (Model bunu kullanacak)")
print(f"Canlı Dosya    (tweets_live.csv):  {len(live_df)} satır. (Producer bunu kullanacak)")

