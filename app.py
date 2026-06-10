%%writefile app.py
import streamlit as st
import pickle
import numpy as np
import re
import math
from urllib.parse import urlparse
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Deep Learning Phishing Website Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Deep Learning Phishing Website Detector")
st.write("Enter a website URL link below to evaluate if it is genuine or a phishing risk.")

# 1. Load trained artifacts safely
@st.cache_resource
def load_resources():
    model = load_model('phishing_detector.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"Error loading system assets: {e}")

# 2. Entropy calculation helper
def calculate_entropy(text):
    if not text:
        return 0
    probabilities = [float(text.count(c)) / len(text) for c in set(text)]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

# 3. Comprehensive 22-Feature Extraction Function
def extract_all_22_features(url):
    features = {}

    # Parse URL parts safely
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    path = parsed.path if parsed.netloc else ""
    query = parsed.query

    # 1. URL Length strings
    features['url_len'] = len(url)

    # 2. Domain Length
    features['dom_len'] = len(domain)

    # 3. Is IP Address present
    ip_pattern = r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
    features['is_ip'] = 1 if re.search(ip_pattern, domain) else 0

    # 4. TLD Length estimation
    tld_match = domain.split('.')[-1] if '.' in domain else ""
    features['tld_len'] = len(tld_match)

    # 5. Subdomain Count
    dot_count_domain = domain.count('.')
    features['subdom_cnt'] = max(0, dot_count_domain - 1) if dot_count_domain > 0 else 0

    # Character counts across the whole URL
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    specials = len(url) - letters - digits

    # 6, 7, 8: Core baseline counts
    features['letter_cnt'] = letters
    features['digit_cnt'] = digits
    features['special_cnt'] = specials

    # 9, 10, 11, 12, 13, 14: Special Character specific allocations
    features['eq_cnt'] = url.count('=')
    features['qm_cnt'] = url.count('?')
    features['amp_cnt'] = url.count('&')
    features['dot_cnt'] = url.count('.')
    features['dash_cnt'] = url.count('-')
    features['under_cnt'] = url.count('_')

    # 15, 16, 17: Structural Ratios
    total_len = len(url) if len(url) > 0 else 1
    features['letter_ratio'] = letters / total_len
    features['digit_ratio'] = digits / total_len
    features['spec_ratio'] = specials / total_len

    # 18. Is HTTPS protocol handled
    features['is_https'] = 1 if url.lower().startswith('https') else 0

    # 19. Slashes present
    features['slash_cnt'] = url.count('/')

    # 20. String Shannon Entropy
    features['entropy'] = calculate_entropy(url)

    # 21, 22: Path and Query Length breakdowns
    features['path_len'] = len(path)
    features['query_len'] = len(query)

    # Explicit ordered mapping array sequence matching your list exactly
    ordered_columns = [
        'url_len', 'dom_len', 'is_ip', 'tld_len', 'subdom_cnt', 'letter_cnt',
        'digit_cnt', 'special_cnt', 'eq_cnt', 'qm_cnt', 'amp_cnt', 'dot_cnt',
        'dash_cnt', 'under_cnt', 'letter_ratio', 'digit_ratio', 'spec_ratio',
        'is_https', 'slash_cnt', 'entropy', 'path_len', 'query_len'
    ]

    return [features[col] for col in ordered_columns]

# 4. Streamlit Interface Code
user_input = st.text_input("Analyze URL Link:", placeholder="https://www.example.com")

if st.button("Run Security Assessment", use_container_width=True):
    if user_input:
        try:
            # Execute feature vector extraction pipeline
            raw_features = extract_all_22_features(user_input)
            scaled_features = scaler.transform([raw_features])

            # Run model prediction
            prediction = model.predict(scaled_features)
            probability = float(prediction[0][0])

            # Metrics drawer dropdown
            with st.expander("🔍 View Trained Pipeline Alignment Data"):
                st.write(f"*Extracted Features Array (22 Dimensions):*")
                st.write(raw_features)
                st.write(f"*Raw Model Sigmoid Probability Score:* {probability:.4f}")

            # Show output based on threshold classification mapping
            if probability > 0.5:
                confidence = probability * 100
                st.error(f"🚨 DANGER: Phishing Detected! ({confidence:.2f}%)")
            else:
                confidence = (1 - probability) * 100
                st.success(f"✅ Safe: This site appears to be Genuine. ({confidence:.2f}%)")

        except Exception as e:
            st.error(f"Pipeline Processing Error: {e}")
    else:
        st.warning("Please paste a target web link to test.")
