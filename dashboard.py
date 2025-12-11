import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pymongo
from collections import Counter

# 1. MongoDB Bağlantısı
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["TwitterDB"]
collection = db["predictions"]

# Grafik Ayarları
plt.style.use('ggplot') # Daha güzel görünüm için stil
fig, ax = plt.subplots(figsize=(10, 6))

def update_graph(i):
    # 2. Veritabanından Son Verileri Çek
    # Tüm veriyi okumak yerine son 1000 tahmini alalım ki trendi görelim
    cursor = collection.find().sort("_id", -1).limit(1000)
    
    predictions = []
    for doc in cursor:
        if "prediction" in doc:
            predictions.append(doc["prediction"])
            
    # 3. Sayım Yap (0.0: Negatif, 1.0: Pozitif)
    counts = Counter(predictions)
    neg_count = counts.get(0.0, 0)
    pos_count = counts.get(1.0, 0)
    
    # 4. Grafiği Temizle ve Yeniden Çiz
    ax.clear()
    
    categories = ['NEGATİF 😡', 'POZİTİF 😊']
    values = [neg_count, pos_count]
    colors = ['#FF4C4C', '#32CD32'] # Kırmızı ve Yeşil
    
    bars = ax.bar(categories, values, color=colors)
    
    # Başlık ve Etiketler
    ax.set_title(f'Canlı Twitter Duygu Analizi (Son {len(predictions)} Tweet)', fontsize=15)
    ax.set_ylabel('Tweet Sayısı', fontsize=12)
    ax.set_ylim(0, max(values) + 10 if values else 10) # Y eksenini dinamik yap
    
    # Çubukların üzerine sayıları yaz
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# 5. Animasyonu Başlat (1000 milisaniyede = 1 saniyede bir güncelle)
ani = FuncAnimation(fig, update_graph, interval=1000, cache_frame_data=False)

print("Dashboard açılıyor... (Kapatmak için pencereyi kapatın)")
plt.tight_layout()
plt.show()
