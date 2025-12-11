import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np


st.set_page_config(
    page_title="Sentiment Pulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    /* Ana Arkaplan */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Kart Tasarımı */
    .kpi-card {
        background-color: #161b22; 
        border-radius: 12px; 
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
        text-align: center; 
        border: 1px solid #30363d;
        transition: transform 0.3s ease;
    }
    .kpi-card:hover { border-color: #00f2c3; transform: translateY(-3px); }
    
    .kpi-title { 
        color: #8b949e; 
        font-size: 12px; 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px; 
    }
    .kpi-value { 
        font-size: 32px; 
        font-weight: 700; 
        color: #f0f6fc; 
    }
    
    /* Grafik Kutuları */
    .chart-box {
        background-color: #161b22; 
        border-radius: 12px; 
        padding: 15px;
        border: 1px solid #30363d; 
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")

try:
    client = init_connection()
    db = client["TwitterDB"]
    collection = db["predictions"]
except:
    st.error("MongoDB Bağlantı Hatası!")
    st.stop()


header_placeholder = st.empty()
kpi_placeholder = st.empty()
charts_placeholder = st.empty()
bottom_placeholder = st.empty()

while True:
    
    total_count_real = collection.count_documents({}) 
    cursor = collection.find().sort("_id", -1).limit(2000)
    data = list(cursor)

    if not data:
        header_placeholder.warning("Sistem başlatılıyor, veri bekleniyor...")
        time.sleep(2)
        continue

    df = pd.DataFrame(data)
    
    
    df = df.sort_values('_id', ascending=True).reset_index(drop=True)
    
    
    df['step'] = df.index + 1

    if "label" in df.columns:
        df['real_val'] = df['label'].apply(lambda x: 1.0 if x == 4 else 0.0)
        df['is_correct'] = (df['real_val'] == df['prediction']).astype(int)
        
       
        df['train_acc'] = df['is_correct'].expanding().mean() * 100
        df['train_loss'] = 1 - df['is_correct'].expanding().mean()
        
       
        df['val_acc'] = df['is_correct'].rolling(window=50, min_periods=1).mean() * 100
        df['val_loss'] = 1 - df['is_correct'].rolling(window=50, min_periods=1).mean()
        
        current_acc = df['train_acc'].iloc[-1]
    else:
        df['train_acc'] = 0; df['val_acc'] = 0
        df['train_loss'] = 0; df['val_loss'] = 0
        current_acc = 0

    pos_count = len(df[df['prediction']==1.0])
    neg_count = len(df[df['prediction']==0.0])

    
    
    with header_placeholder.container():
        c1, c2 = st.columns([6, 1])
        c1.title("⚡ SENTIMENT PULSE")
        c1.markdown("REAL-TIME MODEL TRAINING MONITORING")
        c2.markdown("### 🟢 LIVE")
        st.markdown("---")

    with kpi_placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        def kpi(title, val, color="#ffffff"):
            return f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value' style='color:{color}'>{val}</div></div>"

        c1.markdown(kpi("TOTAL STEPS", f"{total_count_real}", "#ffffff"), unsafe_allow_html=True)
        c2.markdown(kpi("MODEL ACCURACY", f"%{current_acc:.1f}", "#00f2c3"), unsafe_allow_html=True)
        c3.markdown(kpi("POSITIVE", f"{pos_count}", "#1d8cf8"), unsafe_allow_html=True)
        c4.markdown(kpi("NEGATIVE", f"{neg_count}", "#fd5d93"), unsafe_allow_html=True)
        st.markdown("###")

    with charts_placeholder.container():
        col_acc, col_loss = st.columns(2)

        
        with col_acc:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            st.markdown("#### 📈 Accuracy Over Steps")
            
            fig_acc = go.Figure()
            
           
            fig_acc.add_trace(go.Scatter(
                x=df['step'], y=df['train_acc'],
                mode='lines', name='Training',
                line=dict(color='#00f2c3', width=4)
            ))
            
            
            fig_acc.add_trace(go.Scatter(
                x=df['step'], y=df['val_acc'],
                mode='lines', name='Validation',
                line=dict(color='#bd93f9', width=2)
            ))
            
            
            min_y = max(0, min(df['train_acc'].min(), df['val_acc'].min()) - 5)
            max_y = min(100, max(df['train_acc'].max(), df['val_acc'].max()) + 5)
            
            fig_acc.update_layout(
                height=320, margin=dict(l=10,r=10,t=30,b=10),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(range=[min_y, max_y], title="Accuracy %", showgrid=True, gridcolor='#30363d'),
                xaxis=dict(title="Training Steps (Batches)", showgrid=False)
            )
            st.plotly_chart(fig_acc, use_container_width=True, key=f"acc_{time.time()}")
            st.markdown("</div>", unsafe_allow_html=True)

        
        with col_loss:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            st.markdown("#### 📉 Loss Curve")
            
            fig_loss = go.Figure()
            
            
            fig_loss.add_trace(go.Scatter(
                x=df['step'], y=df['train_loss'],
                mode='lines', name='Train Loss',
                line=dict(color='#00f2c3', width=4)
            ))
            
            
            fig_loss.add_trace(go.Scatter(
                x=df['step'], y=df['val_loss'],
                mode='lines', name='Val Loss',
                line=dict(color='#bd93f9', width=2)
            ))
            
            
            max_loss = max(df['train_loss'].max(), df['val_loss'].max()) + 0.1
            
            fig_loss.update_layout(
                height=320, margin=dict(l=10,r=10,t=30,b=10),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(range=[0, max_loss], title="Loss Error", showgrid=True, gridcolor='#30363d'),
                xaxis=dict(title="Training Steps (Batches)", showgrid=False)
            )
            st.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{time.time()}")
            st.markdown("</div>", unsafe_allow_html=True)

    with bottom_placeholder.container():
        col_cm, col_dist = st.columns(2)
        
        with col_cm:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            st.markdown("#### 🟦 Confusion Matrix")
            if "label" in df.columns:
                tp = len(df[(df['prediction'] == 1.0) & (df['real_val'] == 1.0)])
                tn = len(df[(df['prediction'] == 0.0) & (df['real_val'] == 0.0)])
                fp = len(df[(df['prediction'] == 1.0) & (df['real_val'] == 0.0)])
                fn = len(df[(df['prediction'] == 0.0) & (df['real_val'] == 1.0)])
                z = [[tn, fp], [fn, tp]]
                x = ['Pred NEG', 'Pred POS']; y = ['Real NEG', 'Real POS']
                fig_cm = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='RdBu', texttemplate="%{z}", textfont={"size": 20, "color": "white"}, showscale=False))
                fig_cm.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{time.time()}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        with col_dist:
            st.markdown("<div class='chart-box'>", unsafe_allow_html=True)
            st.markdown("#### 🍩 Prediction Distribution")
            fig_pie = px.pie(names=['Positive', 'Negative'], values=[pos_count, neg_count], hole=0.6, color_discrete_sequence=['#1d8cf8', '#fd5d93'])
            fig_pie.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{time.time()}")
            st.markdown("</div>", unsafe_allow_html=True)

    time.sleep(1)
