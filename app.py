from flask import Flask, render_template, request
import pickle
from scipy.sparse import hstack
from preprocessing_utils import clean_text, detect_target

app = Flask(__name__)

# ===========================
# 🔹 Load Models & Encoders
# ===========================

# Binary Classification
try:
    with open('model/model_binary_nb.pkl', 'rb') as f:
        binary_model = pickle.load(f)
except Exception as e:
    print("Gagal load binary model:", e)
    binary_model = None

with open('model/tfidf_vectorizer_binary.pkl', 'rb') as f:
    tfidf_binary = pickle.load(f)

with open('model/target_encoder_binary.pkl', 'rb') as f:
    encoder_binary = pickle.load(f)

# Multiclass Classification
try:
    with open('model/model_multiclass_nb.pkl', 'rb') as f:
        multiclass_model = pickle.load(f)
except Exception as e:
    print("Gagal load model multiclass:", e)
    multiclass_model = None

with open('model/tfidf_vectorizer_multiclass.pkl', 'rb') as f:
    tfidf_multi = pickle.load(f)

with open('model/target_encoder_multiclass.pkl', 'rb') as f:
    encoder_multi = pickle.load(f)

# ===========================
# 🔹 Keterangan Kategori
# ===========================
penjelasan_kategori = {
    'kata kasar': 'Penggunaan kata-kata kotor atau tidak sopan.',
    'ancaman': 'Pernyataan yang menakut-nakuti atau berisi niat menyakiti.',
    'pelecehan': 'Komentar yang merendahkan secara seksual atau pribadi.',
    'body shaming': 'Mengomentari fisik seseorang dengan negatif.',
    'penghinaan': 'Mengejek atau merendahkan orang lain.',
    'Bukan Cyberbullying': 'Kalimat ini tidak mengandung unsur cyberbullying.',
    'Refleksi Diri': 'Kalimat ini mencela diri sendiri, bukan bentuk perundungan terhadap orang lain.'
}

# ===========================
# 🔹 Routes
# ===========================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi():
    kalimat = request.form['kalimat']

    # --- Preprocessing ---
    kalimat_clean = clean_text(kalimat)
    target_kalimat = detect_target(kalimat) or "umum"

    # ===========================
    # 🔍 Deteksi Refleksi Diri
    # ===========================
    refleksi_pronouns = ['aku', 'saya', 'gue', 'gua', 'gw', 'ane', 'beta']
    kata_negatif = ['jelek', 'bodoh', 'hina', 'goblok', 'bego', 'burik', 'kampungan', 'norak', 'dekil', 'tolol', 'gendut', 'gendutan']

    kalimat_tokens = kalimat_clean.lower().split()

    if any(pron in kalimat_tokens for pron in refleksi_pronouns) and any(neg in kalimat_tokens for neg in kata_negatif):
        label_utama = "Refleksi Diri"
        skor_utama = "-"
        confidence_multi = None
        hasil_dummy = {
            "Refleksi Diri": "Kalimat mencela diri sendiri, bukan bentuk perundungan."
        }

        return render_template(
            'index.html',
            kalimat_input=kalimat,
            label_utama=label_utama,
            skor_utama=skor_utama,
            confidence=confidence_multi,
            hasil=hasil_dummy,
            target=None,
            penjelasan_kategori=penjelasan_kategori
        )

    # ===============================
    # 🔸 Tahap 1: Binary Classification
    # ===============================
    X_text_bin = tfidf_binary.transform([kalimat_clean])
    X_target_bin = encoder_binary.transform([[target_kalimat]])
    X_bin = hstack([X_text_bin, X_target_bin])

    probas_bin = binary_model.predict_proba(X_bin)[0]
    label_bin = binary_model.classes_[probas_bin.argmax()]
    confidence_bin = probas_bin.max() * 100

    if label_bin == 0:
        label_utama = "Bukan Cyberbullying"
        skor_utama = f"{confidence_bin:.2f}%"
        confidence_multi = confidence_bin
        hasil_dummy = {
            "Bukan Cyberbullying": f"{confidence_bin:.2f}%"
        }

        return render_template(
            'index.html',
            kalimat_input=kalimat,
            label_utama=label_utama,
            skor_utama=skor_utama,
            confidence=confidence_multi,
            hasil=hasil_dummy,
            target=None,
            penjelasan_kategori=penjelasan_kategori
        )

    # ===============================
    # 🔸 Tahap 2: Multiclass Classification
    # ===============================
    X_text_multi = tfidf_multi.transform([kalimat_clean])
    X_target_multi = encoder_multi.transform([[target_kalimat]])
    X_multi = hstack([X_text_multi, X_target_multi])

    probas_multi = multiclass_model.predict_proba(X_multi)[0]
    label_mapping = multiclass_model.classes_

    hasil_multi = {
        label: f"{prob*100:.2f}%"
        for label, prob in zip(label_mapping, probas_multi)
    }

    dominant_idx = probas_multi.argmax()
    label_utama = label_mapping[dominant_idx]
    skor_utama = f"{probas_multi[dominant_idx] * 100:.2f}%"
    confidence_multi = probas_multi[dominant_idx] * 100

    return render_template(
        'index.html',
        kalimat_input=kalimat,
        label_utama=label_utama,
        skor_utama=skor_utama,
        confidence=confidence_multi,
        hasil=hasil_multi,
        target=target_kalimat,
        penjelasan_kategori=penjelasan_kategori
    )

# ===========================
# 🔹 Run App
# ===========================
if __name__ == '__main__':
    app.run(debug=True)
