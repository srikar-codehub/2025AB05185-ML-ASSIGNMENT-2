"""
Custom CSS styles for the Heart Disease Classification Streamlit App.
Minimalist monochrome theme with Playfair Display and Source Serif fonts.
"""

CUSTOM_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Global styles */
    .stApp {
        font-family: 'Source Serif 4', Georgia, serif;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    h1 {
        font-size: 3rem !important;
        border-bottom: 4px solid #000 !important;
        padding-bottom: 1rem !important;
    }

    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 2px solid #000 !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        border-bottom: none !important;
        background-color: #fff !important;
        color: #000 !important;
        font-family: 'Source Serif 4', serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #000 !important;
        color: #fff !important;
    }

    /* Remove rounded corners from all elements */
    .stButton > button {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        background-color: #000 !important;
        color: #fff !important;
        font-family: 'Source Serif 4', serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 600 !important;
        transition: all 0.1s !important;
    }

    .stButton > button:hover {
        background-color: #fff !important;
        color: #000 !important;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 0 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
    }

    /* DataFrames */
    .stDataFrame {
        border: 2px solid #000 !important;
    }

    /* Radio buttons - horizontal */
    .stRadio > div {
        flex-direction: row !important;
        gap: 1rem !important;
    }

    .stRadio > div > label {
        border: 2px solid #000 !important;
        padding: 0.5rem 1rem !important;
        background: #fff !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 2px solid #000 !important;
        margin: 2rem 0 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        background-color: #fff !important;
        color: #000 !important;
    }

    .stDownloadButton > button:hover {
        background-color: #000 !important;
        color: #fff !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #000 !important;
        border-radius: 0 !important;
        padding: 2rem !important;
    }

    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 0 !important;
        border-left: 4px solid #000 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 0 !important;
    }

    /* Checkbox */
    .stCheckbox {
        border: 1px solid #000 !important;
        padding: 0.5rem !important;
    }
</style>
"""
