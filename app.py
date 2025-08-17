from flask import Flask, render_template, request, jsonify, send_file, url_for, after_this_request
from flask_cors import CORS
from scipy.sparse import hstack
from preprocessing_utils import clean_text, detect_target
import PyPDF2
from docx import Document
import io
import pickle
from datetime import datetime
import os
import subprocess
import re
import hashlib
import logging
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
logging.basicConfig(level=logging.DEBUG)
logging.info("Aplikasi dimulai")

# Load Models & Encoders
try:
    with open('model/model_binary_nb.pkl', 'rb') as f:
        binary_model = pickle.load(f)
except Exception as e:
    logging.error("Gagal load binary model: %s", e)
    binary_model = None

with open('model/tfidf_vectorizer_binary.pkl', 'rb') as f:
    tfidf_binary = pickle.load(f)

with open('model/target_encoder_binary.pkl', 'rb') as f:
    encoder_binary = pickle.load(f)

try:
    with open('model/model_multiclass_nb.pkl', 'rb') as f:
        multiclass_model = pickle.load(f)
except Exception as e:
    logging.error("Gagal load model multiclass: %s", e)
    multiclass_model = None

with open('model/tfidf_vectorizer_multiclass.pkl', 'rb') as f:
    tfidf_multi = pickle.load(f)

with open('model/target_encoder_multiclass.pkl', 'rb') as f:
    encoder_multi = pickle.load(f)

# Keterangan Kategori
penjelasan_kategori = {
    'kata kasar': 'Teks mengandung kata-kata kasar atau umpatan yang bersifat ofensif.',
    'ancaman': 'Teks mengandung ancaman langsung terhadap keselamatan atau keamanan seseorang.',
    'pelecehan': 'Teks mengandung ujaran yang bersifat seksual atau mengganggu secara personal.',
    'body shaming': 'Teks mengandung komentar negatif tentang penampilan fisik seseorang.',
    'penghinaan': 'Teks mengandung kata-kata yang merendahkan atau menghina seseorang.',
    'sara': 'Teks mengandung ujaran yang menyinggung atau mendiskriminasi berdasarkan SARA (Suku, Agama, Ras, dan Antar-golongan).',
    'bukan cyberbullying': 'Teks tidak mengandung unsur cyberbullying atau ujaran kebencian.',
    'refleksi diri': 'Teks mencela diri sendiri, bukan bentuk perundungan terhadap orang lain.',
    'ambiguitas': 'Teks mengandung konten yang sulit dipastikan sebagai cyberbullying karena konteksnya ambigu atau memerlukan interpretasi lebih lanjut.'
}

# List kata negatif untuk deteksi kata cyberbullying
kata_negatif = [
    'jelek', 'bodoh', 'hina', 'goblok', 'bego', 'burik', 'kampungan', 'norak', 'dekil', 
    'tolol', 'gendut', 'gendutan', 'kotor', 'bangsat', 'anjing', 'babi', 'sialan', 
    'brengsek', 'tai', 'najis', 'dungu', 'idiot', 'stupid', 'fat', 'ugly', 'loser',
    'cina', 'jawa', 'sunda', 'papua', 'kafir', 'munafik', 'haram', 'infidel', 'heathen',
    'etnis', 'rasis', 'agama', 'sekte', 'budha', 'hindu', 'kristen', 'islam',
    'mental lu lemah'
]

# In-memory history storage with unique identification
history = []

def extract_cyber_words(kalimat_clean):
    """Extract cyberbullying-related words from cleaned text."""
    tokens = kalimat_clean.lower().split()
    return [word for word in tokens if word in kata_negatif] if tokens else ["Tidak ada kata spesifik terdeteksi"]

def extract_text_from_file(file):
    """Extract text from uploaded file (PDF, DOC, TXT)."""
    filename = file.filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif filename.endswith('.doc'):
            doc = Document(io.BytesIO(file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
    except Exception as e:
        logging.error("Gagal ekstrak teks dari file: %s", e)
        return ""
    return text.strip()

def sanitize_latex(text):
    """Sanitize text for LaTeX by escaping special characters."""
    if not text:
        return ""
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
        '\\': '\\textbackslash{}'
    }
    for char, escape in replacements.items():
        text = text.replace(char, escape)
    return text

def get_unique_id(entry):
    """Generate a unique ID based on tweet and timestamp."""
    entry_str = f"{entry['tweet']}{entry['timestamp']}"
    return hashlib.md5(entry_str.encode()).hexdigest()

@app.route('/')
def home():
    return render_template('index.html', history=history, penjelasan_kategori=penjelasan_kategori)

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi():
    global history
    tweets = []
    kalimat_input = request.form.get('kalimat', '').strip()

    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if file.filename.lower().endswith(('.pdf', '.doc', '.txt')):
            file_text = extract_text_from_file(file)
            if not file_text:
                return render_template('index.html', error="Gagal mengekstrak teks dari file.", kalimat_input=kalimat_input, history=history, penjelasan_kategori=penjelasan_kategori)
            if kalimat_input:
                tweets = [t.strip() for t in kalimat_input.split('\n') if t.strip()]
                file_tweets = [t.strip() for t in file_text.split('\n') if t.strip()]
                tweets.extend(file_tweets)
            else:
                tweets = [t.strip() for t in file_text.split('\n') if t.strip()]
            kalimat_input = kalimat_input + "\n" + file_text if kalimat_input else file_text
        else:
            return render_template('index.html', error="Hanya file .pdf, .doc, atau .txt yang didukung.", kalimat_input=kalimat_input, history=history, penjelasan_kategori=penjelasan_kategori)
    elif kalimat_input:
        tweets = [t.strip() for t in kalimat_input.split('\n') if t.strip()]
    else:
        return render_template('index.html', error="Masukkan tweet atau unggah file.", kalimat_input=kalimat_input, history=history, penjelasan_kategori=penjelasan_kategori)

    results = []
    low_confidence_warning = False

    for kalimat in tweets:
        if not kalimat:
            continue
        kalimat_clean = clean_text(kalimat)
        target_kalimat = detect_target(kalimat) or "umum"
        cyber_words = extract_cyber_words(kalimat_clean)
        logging.debug("Tweet=%s, Cleaned=%s, Cyber Words=%s", kalimat, kalimat_clean, cyber_words)

        # Deteksi Refleksi Diri
        refleksi_pronouns = ['aku', 'saya', 'gue', 'gua', 'gw', 'ane', 'beta']
        kalimat_tokens = kalimat_clean.lower().split()
        if any(pron in kalimat_tokens for pron in refleksi_pronouns) and any(neg in kalimat_tokens for neg in kata_negatif):
            result = {
                'tweet': kalimat,
                'cyber_words': cyber_words,
                'label': 'refleksi diri',
                'score': '-',
                'confidence': 0
            }
            results.append(result)
            unique_id = get_unique_id({**result, 'timestamp': datetime.now()})
            if not any(get_unique_id(h) == unique_id for h in history):
                history.append({**result, 'timestamp': datetime.now()})
            continue

        # Binary Classification
        X_text_bin = tfidf_binary.transform([kalimat_clean])
        X_target_bin = encoder_binary.transform([[target_kalimat]])
        X_bin = hstack([X_text_bin, X_target_bin])
        probas_bin = binary_model.predict_proba(X_bin)[0]
        label_bin = binary_model.classes_[probas_bin.argmax()]
        confidence_bin = probas_bin.max() * 100

        if label_bin == 0:
            result = {
                'tweet': kalimat,
                'cyber_words': cyber_words,
                'label': 'bukan cyberbullying',
                'score': f"{confidence_bin:.2f}%",
                'confidence': confidence_bin
            }
            results.append(result)
            unique_id = get_unique_id({**result, 'timestamp': datetime.now()})
            if not any(get_unique_id(h) == unique_id for h in history):
                history.append({**result, 'timestamp': datetime.now()})
            continue

        # Multiclass Classification
        X_text_multi = tfidf_multi.transform([kalimat_clean])
        X_target_multi = encoder_multi.transform([[target_kalimat]])
        X_multi = hstack([X_text_multi, X_target_multi])
        probas_multi = multiclass_model.predict_proba(X_multi)[0]
        label_mapping = multiclass_model.classes_
        dominant_idx = probas_multi.argmax()
        label_utama = label_mapping[dominant_idx].lower()
        skor_utama = f"{probas_multi[dominant_idx] * 100:.2f}%"
        confidence_multi = probas_multi[dominant_idx] * 100
        logging.debug("Label Mapping=%s, Probas=%s, Dominant Label=%s", label_mapping, probas_multi, label_utama)

        if confidence_multi < 50:
            low_confidence_warning = True

        result = {
            'tweet': kalimat,
            'cyber_words': cyber_words,
            'label': label_utama,
            'score': skor_utama,
            'confidence': confidence_multi
        }
        results.append(result)
        unique_id = get_unique_id({**result, 'timestamp': datetime.now()})
        if not any(get_unique_id(h) == unique_id for h in history):
            history.append({**result, 'timestamp': datetime.now()})

    return render_template(
        'index.html',
        kalimat_input=kalimat_input,
        results=results,
        low_confidence_warning=low_confidence_warning,
        penjelasan_kategori=penjelasan_kategori,
        history=history
    )

@app.route('/generate_pdf_preview', methods=['POST'])
def generate_pdf_preview():
    data = request.get_json()
    day = data.get('day')
    month = data.get('month')
    year = data.get('year')

    # Filter history based on date
    filtered_history = [entry for entry in history if (
        (day == '' or entry['timestamp'].day == int(day)) and
        (month == '' or entry['timestamp'].month == int(month)) and
        (year == '' or entry['timestamp'].year == int(year))
    )]

    if not filtered_history:
        logging.warning("Tidak ada riwayat untuk filter: day=%s, month=%s, year=%s", day, month, year)
        return jsonify({'error': 'Tidak ada riwayat untuk filter yang dipilih.', 'success': False}), 400

    # Limit history entries to avoid complexity
    filtered_history = filtered_history[:50]

    # Generate PDF with reportlab
    temp_dir = 'temp_pdf'
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, 'riwayat_klasifikasi.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # Data for table
    data = [['Tanggal', 'Tweet', 'Kategori', 'Probabilitas']]
    for entry in filtered_history:
        data.append([
            entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
            entry['tweet'],
            entry['label'],
            entry['score']
        ])

    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)

    # Return URL for preview
    preview_url = url_for('serve_pdf', filename='riwayat_klasifikasi.pdf', _external=True)
    return jsonify({'success': True, 'preview_url': preview_url})

@app.route('/serve_pdf/<filename>')
def serve_pdf(filename):
    temp_dir = 'temp_pdf'
    temp_pdf = os.path.join(temp_dir, filename)
    if os.path.exists(temp_pdf):
        try:
            response = send_file(temp_pdf, mimetype='application/pdf', as_attachment=False)
            response.headers['Content-Disposition'] = 'inline; filename=' + filename
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        except Exception as e:
            logging.error("Error serving PDF: %s", e)
            return jsonify({'error': 'Gagal menampilkan PDF: ' + str(e)}), 500
    else:
        logging.error("PDF file not found: %s", temp_pdf)
        return jsonify({'error': 'File PDF tidak ditemukan.'}), 404

@app.route('/unduh_riwayat')
def unduh_riwayat():
    global history
    temp_dir = 'temp_pdf'
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, 'riwayat_klasifikasi.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # Data for table
    data = [['Tanggal', 'Tweet', 'Kategori', 'Probabilitas']]
    for entry in history:
        data.append([
            entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
            entry['tweet'],
            entry['label'],
            entry['score']
        ])

    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)

    # Hapus file setelah diunduh
    @after_this_request
    def remove_file(response):
        try:
            os.remove(pdf_path)
        except Exception as error:
            app.logger.error("Error deleting file", error)
        return response

    return send_file(pdf_path, as_attachment=True, download_name='riwayat_klasifikasi.pdf')

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global history
    try:
        if not history:
            logging.info("Riwayat sudah kosong")
            return jsonify({'success': False, 'error': 'Riwayat sudah kosong.'}), 400
        history = []
        logging.info("Riwayat berhasil dihapus")
        return jsonify({'success': True, 'message': 'Riwayat berhasil dihapus'})
    except Exception as e:
        logging.error("Gagal menghapus riwayat: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)