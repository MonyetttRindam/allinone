import streamlit as st
import subprocess

# Sidebar untuk memilih aplikasi
app_choices = ['Pilih Aplikasi', 'Cats vs Dogs', 'Food', 'Sentiment Analysis']
choice = st.sidebar.selectbox("Pilih Aplikasi", app_choices)

# Fungsi untuk menjalankan masing-masing aplikasi
def run_app(app_name):
    if app_name == 'Cats vs Dogs':
        subprocess.run(['streamlit', 'run', 'catsvsdogs.py'])
    elif app_name == 'Food':
        subprocess.run(['streamlit', 'run', 'food.py'])
    elif app_name == 'Sentiment Analysis':
        subprocess.run(['streamlit', 'run', 'tomlembong.py'])

# Menampilkan aplikasi berdasarkan pilihan
if choice != 'Pilih Aplikasi':
    run_app(choice)
else:
    st.write("Silahkan pilih aplikasi dari sidebar.")
