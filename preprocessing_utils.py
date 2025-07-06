import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi
stop_factory = StopWordRemoverFactory()
original_stopwords = stop_factory.get_stop_words()

# Kecualikan kata-kata subjek agar bisa digunakan untuk deteksi target
subjek = ['aku', 'kamu', 'gue', 'gua', 'gw', 'saya', 'lu', 'lo', 'loe']
stopwords = [w for w in original_stopwords if w not in subjek]

stemmer = StemmerFactory().create_stemmer()

# Fungsi pembersih teks
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # Hapus URL dan mention
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    # Ganti emoji umum dengan teks (bisa ditambah sesuai kebutuhan)
    emoji_map = {
        "❤️": "cinta", "😡": "marah", "😭": "sedih", "😂": "lucu", "😢": "sedih", "👍": "bagus"
    }
    for emoji, meaning in emoji_map.items():
        text = text.replace(emoji, f" {meaning} ")

    # Hapus karakter non-alfabet (kecuali spasi)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Hilangkan spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # Hilangkan stopwords (kecuali kata subjek)
    tokens = [word for word in text.split() if word not in stopwords]

    # Stemming
    return stemmer.stem(' '.join(tokens))

# Deteksi target kalimat: diri sendiri / orang lain / tidak diketahui
def detect_target(text):
    if not isinstance(text, str):
        return "tidak_diketahui"
    
    text = text.lower()
    self_words = ['gue', 'gua', 'gw', 'aku', 'saya']
    other_words = ['lo', 'lu', 'loe', 'kamu']
    tokens = text.split()

    if any(word in tokens for word in self_words):
        return "diri_sendiri"
    elif any(word in tokens for word in other_words):
        return "orang_lain"
    else:
        return "tidak_diketahui"
