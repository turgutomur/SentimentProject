import time
import pandas as pd
from kafka import KafkaProducer
import json


producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')  
)

TOPIC_NAME = 'twitter-sentiment'

print("Producer çalışmaya başladı... Ctrl+C ile durdurabilirsin.")


df = pd.read_csv('tweets_live.csv', encoding='latin-1', header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])


df = df.sample(frac=1).reset_index(drop=True)


try: 
    for index, row in df.iterrows():
        
        data = {
            'text': row['text'],
            'label': int(row['target']) 
        }
        
        
        producer.send(TOPIC_NAME, value=data)
        
        print(f"Gönderildi: {data['text'][:50]}...") 
        
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Producer durduruldu.")
finally:
    producer.close()
