import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download


st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="centered"
)

# CSS kustom dengan desain yang lebih baik
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f9f3e5 0%, #e8f4f8 50%, #f5e9e3 100%);
        background-attachment: fixed;
    }
    .stButton>button {
        background: linear-gradient(to right, #ff9a3c, #ff6b6b);
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2.2rem;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #ff8a2c, #ff5b5b);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 107, 107, 0.4);
    }
    .uploadedFile {
        border-radius: 18px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        overflow: hidden;
        margin: 1.8rem 0;
        border: 3px solid #ffd8a9;
        background-color: white;
    }
    .title {
        color: #5d4037;
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        font-weight: 800;
    }
    .subtitle {
        color: #7fb3d5;
        text-align: center;
        margin-bottom: 2.2rem;
        font-size: 1.2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #fff9e6 0%, #fff4f4 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border-left: 5px solid #ff9a3c;
        margin-bottom: 1.8rem;
        border-right: 1px solid #ffe0b2;
        border-top: 1px solid #ffe0b2;
        border-bottom: 1px solid #ffe0b2;
    }
    .dataset-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #5d4037;
    }
    .footer {
        text-align: center;
        color: #8d6e63;
        margin-top: 2.5rem;
        font-size: 0.9rem;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
    }
    .paw-icon {
        font-size: 1.5rem;
        margin: 0 0.3rem;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff9a3c 0%, #7fb3d5 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown("""
    <div style='text-align: center;'>
        <h1 class="title">🐱 Cats vs Dogs Classifier 🐶</h1>
        <p class="subtitle">Upload gambar kucing atau anjing, dan biarkan AI mengenalinya!</p>
    </div>
""", unsafe_allow_html=True)

# Container utama
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Informasi tentang dataset
with st.expander("📊 Tentang Dataset", expanded=True):
    st.markdown("""
    <div class="dataset-info">
    <h4>📁 Microsoft Cats vs Dogs Dataset</h4>
    <p>Model ini dilatih menggunakan dataset klasik dari Microsoft yang berisi:</p>
    <ul>
        <li>📸 25,000 gambar kucing dan anjing</li>
        <li>🐱 12,500 gambar kucing dari berbagai ras</li>
        <li>🐶 12,500 gambar anjing dari berbagai ras</li>
        <li>🎯 Dataset ini merupakan benchmark populer untuk klasifikasi gambar biner</li>
        <li>🏷️ Setiap gambar berukuran 128x128 piksel dengan label yang jelas</li>
    </ul>
    <p>Dataset ini awalnya dibuat untuk kompetisi machine learning di Kaggle dan 
    telah menjadi standar untuk tugas klasifikasi kucing vs anjing.</p>
    </div>
    """, unsafe_allow_html=True)

# Informasi tambahan
with st.expander("ℹ️ Cara Penggunaan", expanded=False):
    st.markdown("""
    <div class="info-box">
    <h4 style="color: #5d4037;">📋 Cara Penggunaan:</h4>
    <ul>
        <li>Klik tombol "Browse files" untuk mengunggah gambar</li>
        <li>Pastikan gambar menunjukkan kucing atau anjing dengan jelas</li>
        <li>Tunggu sebentar hingga sistem menganalisis gambar</li>
        <li>Lihat hasil prediksi dan tingkat kepercayaan sistem</li>
    </ul>
    
    <h4 style="color: #5d4037;">💡 Tips untuk hasil terbaik:</h4>
    <ul>
        <li>Gunakan gambar dengan hewan menghadap ke depan</li>
        <li>Hindarkan gambar yang blur atau gelap</li>
        <li>Gambar dengan latar belakang sederhana lebih mudah dikenali</li>
        <li>Format yang didukung: JPG, JPEG, PNG</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id="MonyetttRindam/catsvsdogsabil", filename="catsvsdogs.h5")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader(
    "Pilih gambar...", 
    type=["jpg", "jpeg", "png"],
    help="Unggah gambar kucing atau anjing"
)

if uploaded_file is not None:
    #Tampilkan gambar yang diupload
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tampilkan spinner saat memproses
    with st.spinner('Menganalisis gambar...'):
        # Preprocessing
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)[0][0]
        probability_dog = float(prediction)
        probability_cat = 1 - probability_dog
        
        if probability_dog > 0.5:
            label = "Anjing 🐶"
            confidence = probability_dog
            emoji = "🐶"
            color = "#7fb3d5"  # Biru untuk anjing
        else:
            label = "Kucing 🐱"
            confidence = probability_cat
            emoji = "🐱"
            color = "#ff9a3c"  # Oranye untuk kucing

    st.markdown(f'<div class="prediction-box" style="border-left-color: {color}">', unsafe_allow_html=True)
    
    # Tampilkan hasil dengan ikon besar
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<h1 style='text-align: center; font-size: 4.5rem;'>{emoji}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h2 style='color: {color}; margin-bottom: 0.5rem;'>Hasil Prediksi: {label}</h2>", unsafe_allow_html=True)
        st.write(f"**Tingkat kepercayaan: {confidence:.2%}**")
    
    # Tampilkan progress bar untuk kepercayaan
    st.progress(confidence)
    
    # Tampilkan informasi tambahan
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilitas Anjing", f"{probability_dog:.2%}", 
                 delta="Tinggi" if probability_dog > 0.7 else "Sedang" if probability_dog > 0.5 else "Rendah",
                 delta_color="inverse")
    with col2:
        st.metric("Probabilitas Kucing", f"{probability_cat:.2%}", 
                 delta="Tinggi" if probability_cat > 0.7 else "Sedang" if probability_cat > 0.5 else "Rendah")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Saran berdasarkan hasil prediksi
    if confidence > 0.8:
        st.success(f"🎯 Prediksi sangat yakin bahwa ini adalah {label[:-2]} {emoji}")
    elif confidence > 0.6:
        st.info(f"💡 Prediksi cukup yakin bahwa ini adalah {label[:-2]} {emoji}")
    else:
        st.warning("🤔 Prediksi tidak terlalu yakin. Coba gunakan gambar yang lebih jelas.")

# Footer dengan dekorasi
st.markdown("</div>", unsafe_allow_html=True)  # Tutup main-container

st.markdown("""
    <div class="footer">
        <span class="paw-icon">🐾</span>
        Dibuat menggunakan Streamlit dan TensorFlow 
        <span class="paw-icon">🐾</span>
        <br>
        Menggunakan teknologi Deep Learning CNN oleh MonyetttRindam • Dataset dari Microsoft Research
    </div>
""", unsafe_allow_html=True)

link = 'https://catsvsdogs-7nfnnmrzrjmnfeggxssnvk.streamlit.app/'