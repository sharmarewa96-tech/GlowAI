import base64
import hashlib
import html
from io import BytesIO
import os
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

# ------------------------------------------------------------------------------
# LOCAL ASSET HELPERS
# ------------------------------------------------------------------------------
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def _local_img_b64(filename: str) -> str:
    """Return a validated local image data URI, or an embedded SVG placeholder."""
    path = os.path.join(_ASSET_DIR, filename)
    try:
        with Image.open(path) as image:
            image.verify()
        with open(path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        mime = Image.MIME.get(Image.open(path).format, "image/png")
        return f"data:{mime};base64,{encoded}"
    except (OSError, ValueError, TypeError):
        # Inline SVG keeps HTML cards visible even if a local asset is missing.
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='800' viewBox='0 0 640 800'>"
               "<rect width='640' height='800' fill='#f7ebed'/><circle cx='320' cy='290' r='120' fill='#e8cbd0'/>"
               "<path d='M90 760c20-190 130-270 230-270s210 80 230 270' fill='#e8cbd0'/>"
               "<text x='320' y='720' text-anchor='middle' font-family='sans-serif' font-size='24' fill='#884a52'>Glow AI</text></svg>")
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

def _local_img_path(filename: str) -> str:
    """Return an existing, validated asset path for Streamlit image rendering."""
    path = os.path.join(_ASSET_DIR, filename)
    try:
        with Image.open(path) as image:
            image.verify()
        return path
    except (OSError, ValueError, TypeError):
        return os.path.join(_ASSET_DIR, "image_placeholder.png")

def _render_feature_profile(features: dict) -> None:
    """Show measured facial proportions as clear, approximate style descriptors."""
    labels = [
        ("Face shape", "face_shape"),
        ("Brows", "brows"),
        ("Eyes", "eyes"),
        ("Cheekbones", "cheekbones"),
        ("Nose", "nose"),
        ("Lips", "lips"),
    ]
    cards = "".join(
        f"<div style='background:#fff8f8;border:1px solid #ead2d5;border-radius:12px;padding:10px 12px;min-width:0;'>"
        f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#884a52;font-weight:700;'>{html.escape(label)}</div>"
        f"<div style='font-size:13px;font-weight:600;color:#201a1c;margin-top:3px;line-height:1.4;'>{html.escape(str(features.get(key, 'Re-scan portrait to estimate')))}</div></div>"
        for label, key in labels
    )
    st.markdown(
        "<div class='stitch-card' style='margin-top:14px;padding:18px;'>"
        "<span class='stitch-pill stitch-pill-secondary'>FEATURE SHAPE GUIDE</span>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px;margin-top:12px;'>"
        + cards
        + "</div><p style='font-size:10.5px;color:#766b6d;margin:10px 0 0;line-height:1.45;'>"
        "Approximate front-view proportions from facial landmarks. Pose, expression, lighting, and camera perspective can affect these style estimates.</p></div>",
        unsafe_allow_html=True,
    )


def _show_image_inline(image: Image.Image, caption: str = "") -> None:
    """Render PIL images inline to avoid broken Streamlit /media image URLs."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    safe_caption = html.escape(caption)
    safe_alt = html.escape(caption or "Glow AI image", quote=True)
    st.markdown(
        f"""
        <figure style="margin:0; width:100%; padding:12px; border:1px solid #d7c1c3; border-radius:20px; background:#fff; box-shadow:0 8px 24px rgba(136,74,82,0.08); box-sizing:border-box;">
            <img src="{data_uri}" alt="{safe_alt}" style="display:block; width:100%; height:auto; max-height:640px; object-fit:contain; border-radius:12px; background:#f7ebed;" />
            {f'<figcaption style="padding:10px 2px 0; text-align:center; color:#766b6d; font-size:13px;">{safe_caption}</figcaption>' if caption else ''}
        </figure>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 1. PAGE CONFIGURATION & STITCH DESIGN TOKENS
# ==============================================================================
st.set_page_config(
    page_title="Glow AI • Haute Atelier Skincare & Beauty Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "staged_image" not in st.session_state:
    st.session_state.staged_image = None
if "staged_image_source" not in st.session_state:
    st.session_state.staged_image_source = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "last_camera_capture_digest" not in st.session_state:
    st.session_state.last_camera_capture_digest = None
if "camera_analysis_attempted" not in st.session_state:
    st.session_state.camera_analysis_attempted = False
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "mesh_image" not in st.session_state:
    st.session_state.mesh_image = None
if "geometry" not in st.session_state:
    st.session_state.geometry = {}
if "color_data" not in st.session_state:
    st.session_state.color_data = {}
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "active_studio_name" not in st.session_state:
    st.session_state.active_studio_name = "Overview / Studio"
if "selected_skin_type" not in st.session_state:
    st.session_state.selected_skin_type = "combination"
if "selected_concerns" not in st.session_state:
    st.session_state.selected_concerns = ["Barrier Repair", "Dehydration"]
if "budget_tier" not in st.session_state:
    st.session_state.budget_tier = "Curated Mid ($32 – $58)"
for _prefix in ("hair", "eyewear"):
    st.session_state.setdefault(f"{_prefix}_upload_key", 0)
    st.session_state.setdefault(f"{_prefix}_image", None)
    st.session_state.setdefault(f"{_prefix}_source", "")
    if _prefix == "hair":
        st.session_state.setdefault("hair_try_style", "")
for _key, _value in {
    "hair_selected_name": "", "hair_selected_style": "", "hair_comparison": "Side-by-side",
    "eyewear_selected_name": "",
}.items():
    st.session_state.setdefault(_key, _value)
if "virtual_makeup_upload_key" not in st.session_state:
    st.session_state.virtual_makeup_upload_key = 0
if "virtual_makeup_image" not in st.session_state:
    st.session_state.virtual_makeup_image = None
if "virtual_makeup_source" not in st.session_state:
    st.session_state.virtual_makeup_source = ""
for _makeup_key, _makeup_default in {
    "makeup_lipstick": "Rosewood Nude",
    "makeup_blush": "Peach Bloom",
    "makeup_blush_placement": "Lifted sweep",
    "makeup_eyeshadow": "Soft Rose",
    "makeup_eyeliner": "Soft definition",
    "makeup_intensity": 65,
    "makeup_comparison": "Side-by-side",
}.items():
    if _makeup_key not in st.session_state:
        st.session_state[_makeup_key] = _makeup_default
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
else:
    # Remove the old hard-coded demo exchange from sessions that were already open.
    _sample_chat_texts = {
        "Can I use the Niacinamide serum in the morning and Retinal at night together?",
        "Yes! That is an optimal pairing. Niacinamide in the AM reinforces barrier resilience and calms irritation, while evening retinal accelerates cellular turnover without conflict. Remember to follow with SPF 50 during daylight hours.",
    }
    st.session_state.chat_messages = [
        msg for msg in st.session_state.chat_messages
        if not (msg.get("time") == "10:14 AM" and msg.get("text") in _sample_chat_texts)
    ]

def reset_all_visage_state():
    """Resets staged and analyzed visage state cleanly so the user can upload a new photo."""
    st.session_state.analyzed = False
    st.session_state.staged_image = None
    st.session_state.staged_image_source = None
    st.session_state.last_uploaded_name = None
    st.session_state.last_camera_capture_digest = None
    st.session_state.camera_analysis_attempted = False
    st.session_state.original_image = None
    st.session_state.mesh_image = None
    st.session_state.geometry = {}
    st.session_state.color_data = {}
    st.session_state.uploader_key += 1


def navigate_to_studio(studio_name: str) -> None:
    """Keep the sidebar radio and active page in sync for result-card shortcuts."""
    st.session_state.active_studio_name = studio_name
    st.session_state.atelier_sidebar_nav = studio_name

# ==============================================================================
# 2. STITCH "ETHEREAL EDITORIAL BEAUTY" CSS DESIGN SYSTEM
# ==============================================================================
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
/* Exact Stitch Color Tokens */
:root {
    --color-surface: #fff8f8;
    --color-surface-dim: #e3d7da;
    --color-surface-bright: #fff8f8;
    --color-surface-lowest: #ffffff;
    --color-surface-low: #fdf1f3;
    --color-surface-container: #f7ebed;
    --color-surface-high: #f1e5e8;
    --color-surface-highest: #ece0e2;
    --color-on-surface: #201a1c;
    --color-on-surface-variant: #524344;
    --color-primary: #884a52;
    --color-primary-container: #a4616a;
    --color-primary-fixed: #ffd9dc;
    --color-secondary: #7c5357;
    --color-secondary-fixed: #ffdadc;
    --color-on-secondary-fixed: #301216;
    --color-tertiary-fixed: #e8defb;
    --color-on-tertiary-fixed: #1e182d;
    --color-outline: #857374;
    --color-outline-variant: #d7c1c3;
    --shadow-ambient: 0 10px 30px -5px rgba(196, 124, 133, 0.08), 0 4px 12px -2px rgba(36, 30, 32, 0.03);
    --shadow-modal: 0 24px 48px -12px rgba(196, 124, 133, 0.14), 0 8px 24px -4px rgba(36, 30, 32, 0.04);
}
.stApp {
    background-color: var(--color-surface);
    color: var(--color-on-surface);
    font-family: 'Plus Jakarta Sans', sans-serif;
}
/* Typography Rules */
h1, h2, h3, .font-headline {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--color-on-surface);
    letter-spacing: -0.01em;
}
p, span, div, label, input, button {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
/* Left Atelier Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--color-outline-variant) !important;
    box-shadow: 0 10px 30px -5px rgba(196,124,133,0.06);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--color-primary) !important;
}
/* Haute Atelier Logo Monogram */
.atelier-monogram {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-container) 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 700;
    font-size: 20px;
    box-shadow: 0 4px 12px rgba(136, 74, 82, 0.25);
}
/* Stitch Luxury Cards */
.stitch-card {
    background: #ffffff;
    border: 1px solid var(--color-outline-variant);
    border-radius: 24px;
    padding: 24px;
    box-shadow: var(--shadow-ambient);
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stitch-card:hover {
    box-shadow: 0 14px 34px -4px rgba(196, 124, 133, 0.12);
}
/* Pill Badges */
.stitch-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: var(--color-surface-high);
    color: var(--color-primary);
    border: 1px solid var(--color-outline-variant);
}
.stitch-pill-active {
    background: var(--color-primary);
    color: #ffffff !important;
    border-color: var(--color-primary);
    box-shadow: 0 2px 8px rgba(136, 74, 82, 0.25);
}
.stitch-pill-secondary {
    background: var(--color-secondary-fixed);
    color: var(--color-on-secondary-fixed);
    border-color: #f7c5cb;
}
.stitch-pill-tertiary {
    background: var(--color-tertiary-fixed);
    color: var(--color-on-tertiary-fixed);
    border-color: #d8cbeb;
}
/* Hero Section */
.stitch-hero {
    background: linear-gradient(135deg, #ffffff 0%, #fdf1f3 50%, #f7ebed 100%);
    border: 1px solid var(--color-outline-variant);
    border-radius: 28px;
    padding: 28px 32px;
    box-shadow: var(--shadow-ambient);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
/* Primary CTA Buttons */
div.stButton > button {
    background: linear-gradient(135deg, var(--color-primary) 0%, #763b43 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 9999px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.02em;
    min-height: 44px;
    box-shadow: 0 6px 18px rgba(136, 74, 82, 0.25);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #763b43 0%, #5d2b32 100%);
    box-shadow: 0 8px 24px rgba(136, 74, 82, 0.35);
    transform: translateY(-1px);
}
/* Radio & Selectbox Styling */
.stRadio [role="radiogroup"] {
    gap: 8px;
}
/* Top Sticky Bar */
.stitch-topbar {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--color-outline-variant);
    padding: 12px 24px;
    margin: -1.5rem -1.5rem 1.5rem -1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
/* Visage Gallery Card */
.visage-gallery-card {
    border-radius: 20px;
    overflow: hidden;
    background: #ffffff;
    border: 1px solid #d7c1c3;
    box-shadow: 0 4px 14px rgba(196, 124, 133, 0.06);
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}
.visage-gallery-card:hover {
    box-shadow: 0 8px 24px rgba(196, 124, 133, 0.14);
    transform: translateY(-2px);
}
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. COMPLETE STITCH DATASETS (PRODUCTS, ROUTINES, INGREDIENTS, VISAGES)
# ==============================================================================
PRODUCTS_BY_TYPE = {
    "combination": [
        {
            "id": "cb-1",
            "cat": "Face Cleanser",
            "title": "Gentle Hydrating Milk-to-Oil Emulsion",
            "tag": "Bio-Balanced Emulsion",
            "price": "$28 – $36",
            "actives": ["Oat Beta-Glucan", "Plant Squalane", "Centella"],
            "why": "Dissolves impurities without stripping cheek moisture or overloading the T-zone.",
            "ritual": "Massage 1-2 pumps onto dry skin for 60s, emulsify with warm water, rinse gently.",
        },
        {
            "id": "cb-2",
            "cat": "Barrier Serum",
            "title": "Multi-Molecular Hyaluronic & 5% Niacinamide",
            "tag": "Cellular Matrix",
            "price": "$38 – $48",
            "actives": ["Sodium Hyaluronate", "Niacinamide 5%", "Panthenol"],
            "why": "Strengthens lipid matrix, evens skin tone, and delivers multi-depth cellular hydration.",
            "ritual": "Press 3-4 drops onto damp face and neck before applying cream.",
        },
        {
            "id": "cb-3",
            "cat": "Moisturizer",
            "title": "Peptide Barrier Recovery Cloud Cream",
            "tag": "Barrier Recovery",
            "price": "$48 – $58",
            "actives": ["5-Ceramide Complex", "Copper Tripeptide", "Shea Esters"],
            "why": "Nourishing cushion texture provides all-day barrier comfort without pore congestion.",
            "ritual": "Warm pea-sized amount between fingertips; smooth across facial contours upward.",
        },
        {
            "id": "cb-4",
            "cat": "Sunscreen SPF 50+",
            "title": "Invisible Fluid Sheer Mineral Shield SPF 50+",
            "tag": "Mineral Defense",
            "price": "$34 – $44",
            "actives": ["Non-Nano Zinc 12%", "Ectoin", "Green Tea"],
            "why": "100% mineral photoprotection with zero chalky cast and sheer dewy luminous finish.",
            "ritual": "Apply two finger-lengths as final AM step 15 minutes before light exposure.",
        },
        {
            "id": "cb-5",
            "cat": "Lip Treatment",
            "title": "Peptide Ceramide Lip Glaze & Treatment",
            "tag": "Barrier Cushion",
            "price": "$24 – $32",
            "actives": ["Palmitoyl Tripeptide-38", "Shea Butter", "Plant Squalane"],
            "why": "Plumps fine micro-lines and reinforces vermilion border hydration for 12 hours.",
            "ritual": "Glide over bare lips or apply thick layer at bedtime as an overnight cocoon.",
        },
    ],
    "oily": [
        {
            "id": "oil-1",
            "cat": "Face Cleanser",
            "title": "Salicylic Acid & Niacinamide Clarifying Gel",
            "tag": "Sebum Clarifying",
            "price": "$24 – $32",
            "actives": ["Salicylic Acid 2%", "Niacinamide 4%", "Tea Tree"],
            "why": "Penetrates sebum channels to gently dissolve micro-comedones and refine enlarged pores.",
            "ritual": "Lather with lukewarm water on T-zone for 45 seconds; rinse thoroughly.",
        },
        {
            "id": "oil-2",
            "cat": "Pore Serum",
            "title": "BHA Pore Clarifier & Zinc PCA Elixir",
            "tag": "Sebum Equalizer",
            "price": "$32 – $42",
            "actives": ["Betaine Salicylate", "Zinc PCA 1%", "Centella"],
            "why": "Suppresses excess midday shine while calming post-blemish redness.",
            "ritual": "Apply 3 drops morning and evening before gel moisturizer.",
        },
        {
            "id": "oil-3",
            "cat": "Moisturizer",
            "title": "Oil-Free Water Gel Hydra-Quench",
            "tag": "Ultralight Hydration",
            "price": "$36 – $46",
            "actives": ["Micro-Hyaluronic Acid", "Bamboo Water", "Allantoin"],
            "why": "Delivers weightless 24h water reservoir without adding pore-clogging heavy lipids.",
            "ritual": "Smooth generous gel amount across face; absorbs with non-greasy matte finish.",
        },
        {
            "id": "oil-4",
            "cat": "Sunscreen SPF 50",
            "title": "Weightless Mattifying Fluid SPF 50",
            "tag": "Pore-Blur Defense",
            "price": "$32 – $40",
            "actives": ["Silica Microspheres", "Zinc Oxide 10%", "Niacinamide"],
            "why": "Absorbs excess sebum shine throughout the day with undetectable featherlight coverage.",
            "ritual": "Shake bottle well and apply 2 finger lengths evenly across face and neck.",
        },
        {
            "id": "oil-5",
            "cat": "Lip Treatment",
            "title": "Botanical Squalane Featherlight Lip Balm",
            "tag": "Zero-Tack Cushion",
            "price": "$18 – $24",
            "actives": ["Plant Squalane", "Ceramide EOP", "Rosemary"],
            "why": "Weightless conditioning without sticky residue; hydrates lip borders effortlessly.",
            "ritual": "Apply as needed for soft, matte-natural lip comfort.",
        },
    ],
    "dry": [
        {
            "id": "dry-1",
            "cat": "Face Cleanser",
            "title": "Squalane Rich Milky Cleansing Oil & Balm",
            "tag": "Nutrient-Rich Balm",
            "price": "$32 – $40",
            "actives": ["Olive Squalane 20%", "Camellia Seed", "Vitamin E"],
            "why": "Nourishes depleted lipid membranes without stripping skin moisture or stinging.",
            "ritual": "Warm between palms, massage onto dry face, melt away with warm wash cloth.",
        },
        {
            "id": "dry-2",
            "cat": "Hydration Serum",
            "title": "7-Molecular Multi-Hyaluronic Essence",
            "tag": "Deep Dermal Plump",
            "price": "$42 – $52",
            "actives": ["Multi-Weight Hyaluronic", "Beta-Glucan", "Snow Mushroom"],
            "why": "Binds deep moisture in every layer of the epidermis, relieving taut flaking skin.",
            "ritual": "Press 4 drops onto damp skin immediately post-cleanse.",
        },
        {
            "id": "dry-3",
            "cat": "Rich Cream",
            "title": "Ceramide & Phytosterol Deep Moisture Butter",
            "tag": "Lipid Barrier Seal",
            "price": "$52 – $68",
            "actives": ["Ceramides 1, 3, 6-II", "Phytosterols", "Shea Butter"],
            "why": "Repairs micro-fissures and seals moisture in dry skin all day.",
            "ritual": "Press generous amount between palms and smooth over face and neck.",
        },
        {
            "id": "dry-4",
            "cat": "Sunscreen SPF 50+",
            "title": "Dewy Nourishing Mineral Glaze SPF 50+",
            "tag": "Dewy Photoprotection",
            "price": "$36 – $46",
            "actives": ["Zinc Oxide 15%", "Argan Oil", "Vitamin E"],
            "why": "Shields against UV rays while infusing dry skin with glowing luminosity and comfort.",
            "ritual": "Apply generously as final morning step over day moisturizer.",
        },
        {
            "id": "dry-5",
            "cat": "Lip Cocoon",
            "title": "Intense Overnight Peptide Lip Butter",
            "tag": "Deep Lipid Wrap",
            "price": "$26 – $34",
            "actives": ["Tripeptide-1", "Murumuru Butter", "Bisabolol"],
            "why": "Deeply repairs chapped lips and restores plump volume overnight.",
            "ritual": "Coat lips generously before bed; leaves a pillowy velvet shield.",
        },
    ],
    "normal": [
        {
            "id": "norm-1",
            "cat": "Face Cleanser",
            "title": "Botanical Antioxidant Foaming Mousse",
            "tag": "Daily Equilibrium",
            "price": "$26 – $34",
            "actives": ["Green Tea Extract", "Amino Acids", "Cucumber Water"],
            "why": "Maintains optimal 5.5 pH balance while refreshing the complexion thoroughly.",
            "ritual": "Pump once into damp palms, cleanse gently, rinse with cool water.",
        },
        {
            "id": "norm-2",
            "cat": "Radiance Serum",
            "title": "Vitamin C & Ferulic Glow Booster 15%",
            "tag": "Photoprotective Glow",
            "price": "$44 – $56",
            "actives": ["L-Ascorbic Acid 15%", "Ferulic Acid", "Vitamin E"],
            "why": "Brightens subtle dullness, defends against free radicals, enhances skin vitality.",
            "ritual": "Apply 3-4 drops in the morning before moisturizer and sunscreen.",
        },
        {
            "id": "norm-3",
            "cat": "Hydra-Lotion",
            "title": "Silk Dew Hydra-Lotion Emulsion",
            "tag": "Weightless Balance",
            "price": "$40 – $50",
            "actives": ["Rice Bran Ferment", "Hyaluronic Acid", "Squalane"],
            "why": "Gives skin a supple, lit-from-within satin finish that wears beautifully under makeup.",
            "ritual": "Apply 1-2 pumps smoothly across face and neck morning and night.",
        },
        {
            "id": "norm-4",
            "cat": "Sunscreen SPF 50",
            "title": "Daily Radiance Fluid Sheer SPF 50",
            "tag": "Invisible Radiance",
            "price": "$34 – $42",
            "actives": ["Avobenzone Complex", "Niacinamide", "Vitamin C"],
            "why": "Weightless daily sunscreen leaving zero residue and a natural healthy sheen.",
            "ritual": "Smooth over entire face and neck every morning.",
        },
        {
            "id": "norm-5",
            "cat": "Lip Treatment",
            "title": "Rose Glow Tinted Ceramide Oil",
            "tag": "Glossy Glaze",
            "price": "$22 – $28",
            "actives": ["Ceramides", "Damask Rose Oil", "Jojoba"],
            "why": "Hydrates and adds a delicate, flattering flush of healthy rose tone.",
            "ritual": "Swipe on bare lips or over lip pencil as needed.",
        },
    ],
    "sensitive": [
        {
            "id": "sens-1",
            "cat": "Face Cleanser",
            "title": "Centella Asiatica Calming Cleansing Milk",
            "tag": "Hypoallergenic Rinse",
            "price": "$26 – $34",
            "actives": ["Madecassoside", "Colloidal Oat", "Chamomile"],
            "why": "Non-foaming soothing milk that dissolves impurities without triggering flushing.",
            "ritual": "Massage very gently over dry face, tissue or rinse with lukewarm water.",
        },
        {
            "id": "sens-2",
            "cat": "Calming Serum",
            "title": "Colloidal Oat & Beta-Glucan Barrier Serum",
            "tag": "Anti-Redness Complex",
            "price": "$38 – $48",
            "actives": ["Beta-Glucan 3%", "Colloidal Oat", "Allantoin"],
            "why": "Rapidly diminishes stinging, reduces redness, and builds epidermal tolerance.",
            "ritual": "Press 3-4 drops gently onto irritated areas twice daily.",
        },
        {
            "id": "sens-3",
            "cat": "Recovery Balm",
            "title": "Ultra-Calming Barrier Restorative Cream",
            "tag": "Dermal Shield",
            "price": "$44 – $54",
            "actives": ["Ceramides", "Bisabolol", "Ectoin"],
            "why": "Fragrance-free cocoon creating an inert shield against allergens and harsh weather.",
            "ritual": "Warm in palms and gently press onto skin without vigorous rubbing.",
        },
        {
            "id": "sens-4",
            "cat": "Sunscreen SPF 50+",
            "title": "100% Pure Physical Mineral Zinc Screen SPF 50+",
            "tag": "Sensitive Physical Shield",
            "price": "$36 – $46",
            "actives": ["Pure Zinc Oxide 18%", "Licorice Root", "Bisabolol"],
            "why": "No chemical filters; physical reflection prevents warmth-induced redness.",
            "ritual": "Pat gently across face 15 minutes before stepping into daylight.",
        },
        {
            "id": "sens-5",
            "cat": "Lip Salve",
            "title": "Pure Bisabolol & Cica Lip Treatment",
            "tag": "Therapeutic Comfort",
            "price": "$20 – $26",
            "actives": ["Pure Cica", "Bisabolol", "Medical-Grade Squalane"],
            "why": "Soothes cracked, sensitive lip rims with hypoallergenic nourishment.",
            "ritual": "Apply frequently throughout the day and before sleeping.",
        },
    ],
}
ROUTINES_BY_TYPE = {
    "combination": {
        "am": [
            {"num": "01", "name": "Gentle Milky Hydrating Cleanser", "desc": "Lukewarm rinse; preserves acid mantle without stripping.", "note": "60-second rinse"},
            {"num": "02", "name": "Niacinamide 5% & Zinc Barrier Serum", "desc": "Micro-pore refinement and atmospheric defense.", "note": "3-4 drops on damp skin"},
            {"num": "03", "name": "Lightweight Ceramide Water-Cream", "desc": "Rapid-absorption matrix replenishing essential lipids.", "note": "Breathable emulsion"},
            {"num": "04", "name": "Broad-Spectrum Mineral SPF 50 Fluid", "desc": "Non-nano zinc oxide with sheer satin finish, no cast.", "note": "Two finger-lengths"},
        ],
        "pm": [
            {"num": "01", "name": "Botanical Squalane Cleansing Balm", "desc": "Dissolves waterproof SPF and environmental particulates.", "note": "Massage then emulsify"},
            {"num": "02", "name": "Low-Dose Encapsulated Retinal (0.05%)", "desc": "Bio-retinoid accelerating epidermal turnover.", "note": "3 nights / week to start"},
            {"num": "03", "name": "Multi-Peptide Copper Lipid Complex", "desc": "Restores dermal matrix tension during natural sleep.", "note": "Rich serum concentrate"},
            {"num": "04", "name": "Restorative Barrier Night Salve & Glaze", "desc": "Minimizes transepidermal water loss (TEWL) overnight.", "note": "Velvet seal"},
        ],
    },
    "oily": {
        "am": [
            {"num": "01", "name": "BHA Balancing Gel Cleanser", "desc": "Gentle lather clearing nocturnal sebum accumulation.", "note": "45-second wash"},
            {"num": "02", "name": "Zinc PCA & 4% Niacinamide Serum", "desc": "Micro-pore tightening, anti-shine regulation.", "note": "3-4 drops on clean skin"},
            {"num": "03", "name": "Oil-Free Water Gel Hydra-Quench", "desc": "Weightless humectant hydration with non-greasy matte finish.", "note": "Pea-sized layer"},
            {"num": "04", "name": "Weightless Mattifying Fluid SPF 50", "desc": "Silica-infused invisible sunscreen controlling shine.", "note": "Two finger-lengths"},
        ],
        "pm": [
            {"num": "01", "name": "Salicylic Deep Cleansing Gel", "desc": "Clears makeup, sunscreen, and daily airborne grime.", "note": "60-second gentle wash"},
            {"num": "02", "name": "Pore Clarifying Bio-Retinol 0.03%", "desc": "Encapsulated turnover treatment regulating keratinization.", "note": "3 times weekly"},
            {"num": "03", "name": "Centella Soothing Recovery Veil", "desc": "Calms inflammation while maintaining pore permeability.", "note": "Light soothing veil"},
            {"num": "04", "name": "Purifying Spot Clay Essence", "desc": "Targeted application over active blemishes.", "note": "Pinpoint treatment"},
        ],
    },
    "dry": {
        "am": [
            {"num": "01", "name": "Squalane Milky Cleansing Oil", "desc": "Nourishes lipid barrier without water stripping.", "note": "Gentle wipe or rinse"},
            {"num": "02", "name": "7-Molecular Multi-Hyaluronic Essence", "desc": "Binds multi-depth cellular hydration.", "note": "Press into damp skin"},
            {"num": "03", "name": "Ceramide & Phytosterol Rich Butter", "desc": "Restores lipid stratum corneum seal.", "note": "Generous layer"},
            {"num": "04", "name": "Dewy Nourishing Mineral Glaze SPF 50+", "desc": "Hydrating photoprotection with glowing finish.", "note": "Two finger-lengths"},
        ],
        "pm": [
            {"num": "01", "name": "Nourishing Cleansing Balm", "desc": "Melts away makeup while leaving skin supple.", "note": "Warm cloth removal"},
            {"num": "02", "name": "Gentle Granactive Retinoid in Squalane", "desc": "Accelerates cell turnover without dryness.", "note": "2-3 drops"},
            {"num": "03", "name": "Triple Lipid Restorative Complex", "desc": "Replenishes physiological lipid balance.", "note": "Night treatment"},
            {"num": "04", "name": "Overnight Ceramide Cocoon Mask", "desc": "Deep occlusion locking in moisture until morning.", "note": "Sleep pack"},
        ],
    },
    "normal": {
        "am": [
            {"num": "01", "name": "Antioxidant Foaming Mousse", "desc": "Cleanses gently while preserving natural moisture balance.", "note": "Quick refresh"},
            {"num": "02", "name": "Vitamin C & Ferulic Booster 15%", "desc": "Brightens and defends against oxidative stress.", "note": "3-4 drops"},
            {"num": "03", "name": "Silk Dew Hydra-Lotion Emulsion", "desc": "Weightless day hydration under makeup.", "note": "1-2 pumps"},
            {"num": "04", "name": "Daily Radiance Fluid Sheer SPF 50", "desc": "Invisible shield leaving a healthy satin glow.", "note": "Mandatory daily"},
        ],
        "pm": [
            {"num": "01", "name": "Micellar Pre-Cleanse Oil", "desc": "Breaks down daytime pollution and SPF.", "note": "Emulsify with water"},
            {"num": "02", "name": "Multi-Peptide Matrixyl Complex", "desc": "Supports collagen and elasticity overnight.", "note": "Full face & neck"},
            {"num": "03", "name": "Barrier Renewal Night Cream", "desc": "Nourishing replenishment while sleeping.", "note": "Smooth evenly"},
            {"num": "04", "name": "Rose Glow Tinted Ceramide Oil", "desc": "Delicate hydration lock for lips and cheeks.", "note": "Finishing touch"},
        ],
    },
    "sensitive": {
        "am": [
            {"num": "01", "name": "Centella Asiatica Calming Milk", "desc": "Non-foaming soothing milk that calms micro-flushing.", "note": "Splash with lukewarm water"},
            {"num": "02", "name": "Colloidal Oat & Beta-Glucan Serum", "desc": "Relieves tightness, redness, and reactive discomfort.", "note": "Gently press, do not rub"},
            {"num": "03", "name": "Ultra-Calming Barrier Cream", "desc": "Fragrance-free, hypoallergenic lipid barrier veil.", "note": "Light layer"},
            {"num": "04", "name": "100% Pure Physical Zinc Screen SPF 50+", "desc": "Zero chemical filters; pure physical sun barrier.", "note": "Two finger-lengths"},
        ],
        "pm": [
            {"num": "01", "name": "Ultra-Gentle Cleansing Emulsion", "desc": "Dissolves impurities with zero friction.", "note": "Tissue or rinse"},
            {"num": "02", "name": "Madecassoside Repair Ampoule", "desc": "Concentrated botanical active for barrier repair.", "note": "Spot or full face"},
            {"num": "03", "name": "Ectoin & Squalane Calming Balm", "desc": "Soothes irritated patches and reinforces resistance.", "note": "Night seal"},
            {"num": "04", "name": "Pure Bisabolol Lip Treatment", "desc": "Hypoallergenic lip hydration wrap.", "note": "Sleep balm"},
        ],
    },
}
INGREDIENTS_DATA = [
    {
        "name": "Niacinamide (Vitamin B3)",
        "tag": "Essential Active",
        "benefits": "Sebum regulation, lipid synthesis, redness & pore reduction.",
        "types": ["Combination", "Oily", "Sensitive"],
        "synergy": "Pairs harmoniously with Hyaluronic Acid, Ceramides & Centella.",
        "badge": "stitch-pill-secondary",
    },
    {
        "name": "Hyaluronic Acid (Multi-Mol)",
        "tag": "Dermal Hydrator",
        "benefits": "High-capacity water binding (up to 1,000x its molecular weight).",
        "types": ["All Skin Types", "Dry", "Dehydrated"],
        "synergy": "Pat onto damp skin for maximum molecular uptake before creams.",
        "badge": "stitch-pill-tertiary",
    },
    {
        "name": "Ceramides (NP, AP, EOP)",
        "tag": "Lipid Stratum",
        "benefits": "Restores intercellular lipid stratum corneum barrier; locks moisture.",
        "types": ["Dry", "Compromised", "Barrier Repair"],
        "synergy": "Crucial companion active when using retinoids or AHA exfoliants.",
        "badge": "stitch-pill-secondary",
    },
    {
        "name": "Centella Asiatica (Cica)",
        "tag": "Vascular Calming",
        "benefits": "Potent soothing madecassoside active; relieves reactive micro-flushing.",
        "types": ["Sensitive", "Reactive", "Post-Sun"],
        "synergy": "Relieves erythema and accelerates natural barrier replenishment.",
        "badge": "stitch-pill-secondary",
    },
    {
        "name": "Salicylic Acid (BHA 2%)",
        "tag": "Follicle Clarifier",
        "benefits": "Lipid-soluble; clears follicular pore debris and reduces micro-comedones.",
        "types": ["Oily", "Congested"],
        "synergy": "Avoid simultaneous layering with strong retinoids on the same night.",
        "badge": "stitch-pill",
    },
    {
        "name": "Squalane (Biomimetic Olive)",
        "tag": "Biomimetic Lipid",
        "benefits": "100% skin-compatible emollient offering non-comedogenic elasticity.",
        "types": ["Universal", "Dry", "Dehydrated"],
        "synergy": "Ultra-stable lipid that will not oxidize or clog pores on the skin surface.",
        "badge": "stitch-pill-tertiary",
    },
]

# Local, bundled image assets for gallery muses and rituals
DIVERSE_VISAGES = [
    {
        "look": "Luminous Glass Dew",
        "hairstyle": "Curtain Bangs & Soft Layers",
        "makeup": "Dewy Glaze, Nude Gloss Lip",
        "demographic": "East Asian Visage",
        "undertone_hint": "Cool-to-Neutral Porcelain",
        "tag": "Glass Dew",
        "skin_rec": "combination",
        "img_file": _local_img_b64("camille.png"),
    },
    {
        "look": "Golden Olive Berry",
        "hairstyle": "Voluminous Side Part Waves",
        "makeup": "Warm Bronze Contour, Glossy Rose Lip",
        "demographic": "South Asian Visage",
        "undertone_hint": "Warm Olive / Golden",
        "tag": "Warm Olive",
        "skin_rec": "combination",
        "img_file": _local_img_b64("model_1.png"),
    },
    {
        "look": "Silver Silk Radiance",
        "hairstyle": "Textured Silver Chin Bob",
        "makeup": "Plump Ceramide Base, Soft Peach Flush",
        "demographic": "Mature Adult (45+)",
        "undertone_hint": "Neutral Rosy Alabaster",
        "tag": "Ageless Glow",
        "skin_rec": "dry",
        "img_file": _local_img_b64("model_2.png"),
    },
    {
        "look": "Sculpted Deep Luster",
        "hairstyle": "Sleek High Architectural Bun",
        "makeup": "High Cheekbone Contour, Cocoa Gloss",
        "demographic": "Deep Rich Visage",
        "undertone_hint": "Rich Espresso / Warm Chestnut",
        "tag": "Melanin Luster",
        "skin_rec": "dry",
        "img_file": _local_img_b64("model_3.png"),
    },
    {
        "look": "Warm Honey Golden",
        "hairstyle": "Loose Flowing Beach Waves",
        "makeup": "Sunlit Champagne Sheen, Honey Nude Lip",
        "demographic": "Warm Bronze Visage",
        "undertone_hint": "Honey Golden Radiance",
        "tag": "Sunlit Sheen",
        "skin_rec": "normal",
        "img_file": _local_img_b64("model_4.png"),
    },
]

_IMG_DOCTOR = _local_img_b64("dr_elise.png")
_IMG_STEP_1 = _local_img_b64("ritual_1.png")
_IMG_STEP_2 = _local_img_b64("ritual_2.png")
# Unsplash photo by Goddessance, free under the Unsplash License.
_IMG_STEP_3 = _local_img_b64("barrier_cream.jpg")

# ==============================================================================
# 4. COMPUTER VISION & CORE PYTHON LOGIC (100% PRESERVED)
# ==============================================================================
def generate_sample_portrait():
    """Generates an elegant synthetic portrait that MediaPipe reliably detects for instant demo testing."""
    img = Image.new("RGB", (480, 580), color=(253, 248, 244))
    draw = ImageDraw.Draw(img)
    draw.ellipse([110, 60, 370, 480], fill=(62, 40, 32))
    draw.ellipse(
        [135, 110, 345, 470],
        fill=(240, 204, 185),
        outline=(218, 175, 155),
        width=2,
    )
    draw.chord([125, 75, 355, 330], 190, 350, fill=(72, 48, 38))
    draw.arc([165, 215, 225, 235], 195, 345, fill=(65, 42, 33), width=3)
    draw.arc([255, 215, 315, 235], 195, 345, fill=(65, 42, 33), width=3)
    draw.ellipse(
        [170, 235, 220, 262],
        fill=(255, 255, 255),
        outline=(85, 55, 45),
        width=1,
    )
    draw.ellipse([186, 240, 204, 258], fill=(68, 45, 34))
    draw.ellipse([192, 244, 198, 250], fill=(255, 255, 255))
    draw.ellipse(
        [260, 235, 310, 262],
        fill=(255, 255, 255),
        outline=(85, 55, 45),
        width=1,
    )
    draw.ellipse([276, 240, 294, 258], fill=(68, 45, 34))
    draw.ellipse([282, 244, 288, 250], fill=(255, 255, 255))
    draw.line(
        [(240, 260), (236, 312), (244, 312)], fill=(208, 160, 140), width=2
    )
    draw.ellipse([155, 290, 205, 330], fill=(245, 188, 182))
    draw.ellipse([275, 290, 325, 330], fill=(245, 188, 182))
    draw.chord([195, 352, 285, 396], 0, 180, fill=(210, 105, 122))
    draw.chord([195, 362, 285, 385], 180, 360, fill=(195, 90, 108))
    return img

@st.cache_data(show_spinner=False, max_entries=4)
def _cached_makeup_face_points(image_bytes: bytes):
    """Detect a face once per uploaded image and cache its refined FaceMesh points."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(image)
    face_mesh_module = mp.solutions.face_mesh
    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.45,
    ) as detector:
        result = detector.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = rgb.shape[:2]
    return [[lm.x * width, lm.y * height] for lm in result.multi_face_landmarks[0].landmark]


def _hex_rgb(color: str):
    value = color.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _landmark_polygon_mask(shape, points, indices, blur_fraction=0.006):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray([points[index] for index in indices], dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    sigma = max(1.0, min(width, height) * blur_fraction)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32) / 255.0


def _blend_cosmetic_tint(rgb, mask, color, strength, saturation_boost=1.0, darken=0.0):
    """Tint a feathered mask in HSV while retaining the source image's shading."""
    if strength <= 0 or not np.any(mask > 0.01):
        return rgb
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[list(_hex_rgb(color))]]), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    alpha = np.clip(mask * strength, 0.0, 0.92)
    source_hue = hsv[:, :, 0]
    delta = ((target[0] - source_hue + 90.0) % 180.0) - 90.0
    hsv[:, :, 0] = (source_hue + delta * alpha) % 180.0
    desired_saturation = np.maximum(hsv[:, :, 1], target[1] * saturation_boost)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - alpha) + desired_saturation * alpha
    hsv[:, :, 2] *= 1.0 - alpha * darken
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def _virtual_makeup_preview(image, points, lipstick, blush, blush_placement, eyeshadow, eyeliner, intensity):
    """Apply soft, feature-aligned lipstick, blush, eyeshadow, and optional liner."""
    source = np.asarray(image.convert("RGB")).copy()
    height, width = source.shape[:2]
    points = np.asarray(points, dtype=np.float32)
    level = float(intensity) / 100.0

    outer_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    inner_lip = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    lip_mask = np.zeros((height, width), dtype=np.uint8)
    for contour in (outer_lip,):
        polygon = np.round(points[contour]).astype(np.int32)
        cv2.fillPoly(lip_mask, [polygon], 255, lineType=cv2.LINE_AA)
    inner_polygon = np.round(points[inner_lip]).astype(np.int32)
    cv2.fillPoly(lip_mask, [inner_polygon], 0)
    lip_sigma = max(1.0, min(width, height) * 0.003)
    lip_mask = cv2.GaussianBlur(lip_mask, (0, 0), lip_sigma).astype(np.float32) / 255.0
    source = _blend_cosmetic_tint(source, lip_mask, lipstick, 0.78 * level, saturation_boost=0.88, darken=0.08)

    # Cheek centers use FaceMesh cheek-region landmarks; placement changes their position.
    cheek_centers = [
        points[117] * 0.62 + points[50] * 0.38,
        points[346] * 0.62 + points[280] * 0.38,
    ]
    face_length = max(float(np.linalg.norm(points[10] - points[152])), 1.0)
    face_width = max(float(np.linalg.norm(points[234] - points[454])), 1.0)
    if blush_placement == "Lifted sweep":
        cheek_centers[0] = cheek_centers[0] * 0.72 + points[234] * 0.28
        cheek_centers[1] = cheek_centers[1] * 0.72 + points[454] * 0.28
        cheek_centers[0][1] -= face_length * 0.025
        cheek_centers[1][1] -= face_length * 0.025
    elif blush_placement == "Soft drape":
        cheek_centers[0][1] += face_length * 0.018
        cheek_centers[1][1] += face_length * 0.018
    blush_mask = np.zeros((height, width), dtype=np.uint8)
    radius_x = max(10, int(face_width * (0.14 if blush_placement == "Soft apples" else 0.16)))
    radius_y = max(7, int(face_length * 0.055))
    cheek_axis = points[454] - points[234]
    cheek_angle = float(np.degrees(np.arctan2(cheek_axis[1], cheek_axis[0])))
    for center in cheek_centers:
        cv2.ellipse(
            blush_mask,
            tuple(np.round(center).astype(int)),
            (radius_x, radius_y),
            cheek_angle,
            0,
            360,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
    blush_mask = cv2.GaussianBlur(blush_mask, (0, 0), max(2.0, min(width, height) * 0.012)).astype(np.float32) / 255.0
    source = _blend_cosmetic_tint(source, blush_mask, blush, 0.68 * level, saturation_boost=0.82, darken=0.02)

    upper_eye_paths = ([33, 246, 161, 160, 159, 158, 157, 173, 133], [263, 466, 388, 387, 386, 385, 384, 398, 362])
    shadow_mask = np.zeros((height, width), dtype=np.uint8)
    band_offset = max(4.0, face_length * 0.045)
    for path in upper_eye_paths:
        lid = points[path].copy()
        raised = lid.copy()
        raised[:, 1] -= band_offset
        band = np.vstack((lid, raised[::-1]))
        cv2.fillPoly(shadow_mask, [np.round(band).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), max(1.0, min(width, height) * 0.004)).astype(np.float32) / 255.0
    source = _blend_cosmetic_tint(source, shadow_mask, eyeshadow, 0.48 * level, saturation_boost=0.66, darken=0.04)

    if eyeliner != "None":
        liner_mask = np.zeros((height, width), dtype=np.uint8)
        thickness = max(2, int(face_width * 0.006))
        for path, outer_index, inner_index in ((upper_eye_paths[0], 33, 133), (upper_eye_paths[1], 263, 362)):
            curve = np.round(points[path]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(liner_mask, [curve], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
            if eyeliner == "Classic wing":
                outer = points[outer_index]
                inner = points[inner_index]
                outward = outer - inner
                norm = max(float(np.linalg.norm(outward)), 1.0)
                wing = outer + outward / norm * (face_width * 0.045)
                wing[1] -= face_length * 0.025
                cv2.line(liner_mask, tuple(np.round(outer).astype(int)), tuple(np.round(wing).astype(int)), 255, thickness=thickness, lineType=cv2.LINE_AA)
        liner_mask = cv2.GaussianBlur(liner_mask, (0, 0), max(0.8, thickness * 0.55)).astype(np.float32) / 255.0
        source = _blend_cosmetic_tint(source, liner_mask, "#34232B", 0.92 * level, saturation_boost=0.9, darken=0.12)

    return Image.fromarray(source)


def _reset_virtual_makeup():
    st.session_state.virtual_makeup_image = None
    st.session_state.virtual_makeup_source = ""
    st.session_state.virtual_makeup_upload_key += 1
    st.session_state.makeup_lipstick = "Rosewood Nude"
    st.session_state.makeup_blush = "Peach Bloom"
    st.session_state.makeup_blush_placement = "Lifted sweep"
    st.session_state.makeup_eyeshadow = "Soft Rose"
    st.session_state.makeup_eyeliner = "Soft definition"
    st.session_state.makeup_intensity = 65
    st.session_state.makeup_comparison = "Side-by-side"



@st.cache_data(show_spinner=False, max_entries=4)
def _cached_style_face_points(image_bytes: bytes):
    """Detect refined FaceMesh landmarks locally for portrait styling."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(image)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.45) as detector:
        result = detector.process(rgb)
    if not result.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return [[lm.x * w, lm.y * h] for lm in result.multi_face_landmarks[0].landmark]


def _estimate_style_face_shape(points):
    p = np.asarray(points, dtype=np.float32)
    ratio = float(np.linalg.norm(p[10] - p[152]) / max(np.linalg.norm(p[234] - p[454]), 1.0))
    jaw = float(np.linalg.norm(p[172] - p[397]) / max(np.linalg.norm(p[234] - p[454]), 1.0))
    upper = float(np.linalg.norm(p[54] - p[284]) / max(np.linalg.norm(p[234] - p[454]), 1.0))
    if ratio >= 1.36: return "Oblong / Long"
    if jaw < 0.78 and upper > 1.02: return "Heart"
    if jaw < 0.76: return "Diamond"
    if ratio <= 1.08 and jaw >= 0.88: return "Square"
    if ratio <= 1.08: return "Round"
    return "Oval / Balanced"



def _face_shape_explanation(shape):
    return {
        "Round":"A softly curved outline with similar face length and width; styling can add gentle vertical or angular contrast.",
        "Square":"A broad outline with a more defined jaw; softly rounded details can balance the angles.",
        "Oblong / Long":"A longer face proportion; lateral movement and frame height can add visual balance.",
        "Heart":"A broader upper face with a narrower lower outline; lighter shapes near the temples keep proportions balanced.",
        "Diamond":"Cheekbones read widest, with a narrower forehead and jaw; softer lines can complement the cheek area.",
        "Oval / Balanced":"Face length and width appear balanced, leaving room to explore a wide range of proportions.",
    }.get(shape,"An approximate front-view proportion estimate. Pose, lighting, and expression can affect the result.")


def _style_recommendations(shape, category):
    """Return exactly three informative, face-shape-specific options with photo assets."""
    key=shape.lower()
    if category=="hair":
        # Existing local portrait photos represent three distinct, real hair silhouettes.
        if "round" in key:
            return [
                ("Long Face-Framing Layers","The longer, softly parted outline adds a lengthening line while keeping movement around the cheeks.","Long","Loose, face-framing waves with blended layers.","Moderate","model_5.png","long"),
                ("Shoulder-Length Soft Waves","A side part and movement below the cheekbones create a light diagonal and balanced width.","Shoulder length","Airy, brushed-out waves with a side part.","Moderate","model_3.png","waves"),
                ("Soft Low Chignon","A clean, low gathered shape keeps the crown from feeling overly tall and leaves the jawline open.","Updo","Polished low bun with soft face-framing tendrils.","Low","model_2.png","chignon"),
            ]
        if "oblong" in key or "long" in key:
            return [
                ("Shoulder-Length Soft Waves","Side volume and a shoulder-grazing finish add width and soften a longer outline.","Shoulder length","Loose waves with fullness through the sides.","Moderate","model_3.png","waves"),
                ("Long Butterfly Layers","Wide, feathered face-framing layers add movement without pulling the silhouette upward.","Long","Long layers with softly flipped, airy ends.","Moderate","model_5.png","long"),
                ("Soft Low Chignon","A low, softly textured updo keeps the silhouette balanced and avoids extra height at the crown.","Updo","Low chignon with relaxed tendrils around the face.","Low","model_2.png","chignon"),
            ]
        if "heart" in key or "diamond" in key:
            return [
                ("Collarbone Face-Framing Layers","Movement that begins around the jaw can balance a narrower lower face while keeping the cheek area light.","Long to collarbone","Soft layers with gentle bends through the ends.","Moderate","model_5.png","long"),
                ("Shoulder-Length Soft Waves","A side-swept part and soft waves create a flattering diagonal around the cheekbones.","Shoulder length","Loose side-parted waves with a light finish.","Moderate","model_3.png","waves"),
                ("Soft Low Chignon","A relaxed low updo lets the cheekbones show while keeping the temple area soft.","Updo","Low bun with fine tendrils and soft texture.","Low","model_2.png","chignon"),
            ]
        return [
            ("Long Butterfly Layers","Feathered lengths add movement around balanced proportions and draw attention to the cheekbones.","Long","Long, flowing layers with softly waved ends.","Moderate","model_5.png","long"),
            ("Shoulder-Length Soft Waves","A versatile shoulder-grazing silhouette frames the eyes while keeping natural proportions visible.","Shoulder length","Brushed-out waves with a soft side part.","Moderate","model_3.png","waves"),
            ("Soft Low Chignon","A polished low updo offers an elegant alternative that highlights the face without excess crown height.","Updo","Relaxed low bun with gentle face-framing pieces.","Low","model_2.png","chignon"),
        ]
    if "round" in key:
        return [
            ("Angular Rectangle","Crisp outer corners bring definition and contrast to rounded contours.","Rectangle · full acetate rim","Espresso, tortoise, or rose crystal","Rectangle"),
            ("Lifted Cat-Eye","An upswept outer edge creates a flattering diagonal and draws attention toward the temples.","Cat-eye · sculpted upper rim","Burgundy, champagne, or black","Cat-eye"),
            ("Transparent Square","A defined lens shape gives gentle structure while pale acetate keeps the look light.","Square · clear acetate","Blush crystal, smoke, or clear","Transparent"),
        ]
    if "square" in key:
        return [
            ("Soft Oval","Rounded edges soften a defined jaw with an easy curved line.","Oval · fine rounded rim","Blush, champagne, or honey","Oval"),
            ("Classic Aviator","Curved lenses offset angular lines and bring a balanced, relaxed silhouette.","Aviator · double bridge","Gold, bronze, or soft silver","Aviator"),
            ("Rounded Cat-Eye","A soft upsweep adds definition without repeating the jaw's straight edges.","Cat-eye · light acetate","Rose crystal, tortoise, or plum","Cat-eye"),
        ]
    if "oblong" in key or "long" in key:
        return [
            ("Tall Aviator","A deeper lens adds visual width and balances a longer face proportion.","Aviator · deep teardrop lens","Gold, bronze, or tortoise","Aviator"),
            ("Wide Oval","A broad oval frame creates a soft horizontal line across the face.","Oval · wide lens fit","Champagne, olive, or clear","Oval"),
            ("Statement Square","A taller square lens adds presence and keeps the frame in proportion.","Square · substantial lens","Tortoise, burgundy, or black","Square"),
        ]
    if "heart" in key or "diamond" in key:
        return [
            ("Subtle Cat-Eye","A fine upswept rim follows the cheekbones without adding a heavy line at the brow.","Cat-eye · fine lower profile","Rose gold, clear, or soft plum","Cat-eye"),
            ("Oval Wire Frame","A narrow rounded shape keeps the upper face feeling open.","Oval · lightweight profile","Champagne, silver, or pale gold","Oval"),
            ("Lightweight Aviator","A curved lens and delicate bridge complement defined cheekbones without a heavy upper rim.","Aviator · slender double bridge","Gold with tea, smoke, or clear lenses","Aviator"),
        ]
    return [
        ("Modern Cat-Eye","A gentle outer lift emphasizes the cheekbones and adds a graceful diagonal.","Cat-eye · medium acetate rim","Tortoise, berry, or rose crystal","Cat-eye"),
        ("Soft Rectangle","A clean horizontal line frames the eyes; eased corners keep the look approachable.","Rectangle · rounded corners","Honey, espresso, or blush crystal","Rectangle"),
        ("Classic Round","A rounded frame adds relaxed contrast to balanced proportions.","Round · slim full rim","Gold, clear, or dark tortoise","Round"),
    ]


_EYEWEAR_PRODUCT_IMAGE_FILES={
    "Cat-eye":"cat_eye.jpg",
    "Rectangle":"rectangle.jpg",
    "Round":"round.jpg",
    "Oval":"oval.jpg",
    "Aviator":"aviator.webp",
    "Square":"square.jpg",
    "Transparent":"square.jpg",
}


def _reference_photo(asset_name):
    """Load a local photographic style reference for the hairstyle studio."""
    path=_local_img_path(asset_name)
    image=ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((900,900),Image.Resampling.LANCZOS)
    return image


def _eyewear_product_image(frame_style):
    """Load an isolated, local product photo without a model or face overlay."""
    filename=_EYEWEAR_PRODUCT_IMAGE_FILES.get(frame_style)
    try:
        if not filename:
            raise FileNotFoundError(f"No product photo is mapped for {frame_style}.")
        path=os.path.join(_ASSET_DIR,"eyewear_products",filename)
        with Image.open(path) as source:
            image=ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1400,900),Image.Resampling.LANCZOS)
        return image
    except (OSError,ValueError,TypeError,FileNotFoundError):
        image=Image.new("RGB",(1200,680),(249,242,242))
        draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((24,24,1176,656),radius=28,fill=(255,255,255),outline=(224,197,201),width=3)
        draw.text((600,320),f"{frame_style} product photo unavailable",fill=(136,74,82),anchor="mm")
        return image


_EYEWEAR_FRAME_INFO={
    "Cat-eye":(
        "A medium-width acetate frame with softly lifted outer corners and a graceful upswept silhouette.",
        "Let the lifted corners be the focal point; pair with simple earrings and softly defined brows.",
    ),
    "Rectangle":(
        "A softly rounded rectangular silhouette with a clean horizontal line and an easy everyday fit.",
        "Try relaxed textures or rounded accessories to balance the frame's neat, horizontal lines.",
    ),
    "Round":(
        "A classic rounded frame with a slim full rim, balanced bridge, and lightweight profile.",
        "Pair the curved lenses with tailored lines or angular accessories for a little contrast.",
    ),
    "Oval":(
        "A softly elongated oval profile with a lightweight rim and gentle, versatile curves.",
        "Keep the look airy with delicate jewelry and soft, clean lines around the neckline.",
    ),
    "Aviator":(
        "A deep curved lens with a refined double bridge and slender metal lines.",
        "Wear with simple shapes and let the double bridge add a subtle architectural detail.",
    ),
    "Square":(
        "A structured square frame with gently softened corners and a modern full-rim profile.",
        "Balance the defined corners with soft fabrics or rounded jewelry for an easy finish.",
    ),
    "Transparent":(
        "A defined square silhouette in clear acetate that gives structure while keeping the look light.",
        "Use the clear frame as a quiet base for a colorful lip, earrings, or a textured outfit.",
    ),
}


def _reset_style_studio(prefix):
    st.session_state[f"{prefix}_image"] = None
    st.session_state[f"{prefix}_source"] = ""
    st.session_state[f"{prefix}_selected_name"] = ""
    st.session_state[f"{prefix}_upload_key"] += 1
    if prefix == "hair":
        st.session_state.hair_try_style = ""
        st.session_state.hair_selected_style = ""
        st.session_state.hair_selected_name = ""
        st.session_state.hair_comparison = "Side-by-side"
    else:
        st.session_state.eyewear_selected_name = ""


def analyze_portrait_landmarks(pil_image):
    """Executes facial landmark detection, mathematical geometry, and CIELAB skin color analysis.
    Preserves 100% of existing geometry and optical analysis logic.
    """
    img_copy = pil_image.copy()
    img_copy.thumbnail((1200, 1200))
    rgb = np.array(img_copy)
    height, width = rgb.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.45,
    ) as face_mesh:
        result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None, "No face detected. Please try a well-lit, front-facing portrait."
    landmarks = result.multi_face_landmarks[0].landmark
    points = np.array([[lm.x * width, lm.y * height] for lm in landmarks])
    # Wireframe mesh in Stitch rose
    mesh_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(mesh_image)
    for a, b in mp_face_mesh.FACEMESH_TESSELATION:
        x1, y1 = points[a].astype(int)
        x2, y2 = points[b].astype(int)
        draw.line((x1, y1, x2, y2), fill=(136, 74, 82), width=1)
    # Geometry calculations
    forehead, chin = points[10], points[152]
    left_cheek, right_cheek = points[234], points[454]
    face_length = float(np.linalg.norm(chin - forehead))
    face_width = float(np.linalg.norm(right_cheek - left_cheek))
    if face_width <= 0 or face_length <= 0:
        return None, "Could not compute reliable proportions from image geometry."
    length_width_ratio = face_length / face_width
    brow_y = float(np.mean(points[[105, 334], 1]))
    nose_base_y = float(np.mean(points[[2, 98, 327], 1]))
    upper = abs(brow_y - forehead[1])
    middle = abs(nose_base_y - brow_y)
    lower = abs(chin[1] - nose_base_y)
    total_thirds = upper + middle + lower
    upper_pct = (upper / total_thirds) * 100 if total_thirds > 0 else 0
    middle_pct = (middle / total_thirds) * 100 if total_thirds > 0 else 0
    lower_pct = (lower / total_thirds) * 100 if total_thirds > 0 else 0
    distance = lambda a, b: float(np.linalg.norm(points[a] - points[b]))
    jaw_width = distance(172, 397)
    upper_face_width = distance(54, 284)
    jaw_to_cheek = jaw_width / face_width
    upper_to_cheek = upper_face_width / face_width
    if length_width_ratio >= 1.36:
        face_shape = "Oblong / Long"
        shape_description = "The face reads longer than it is wide, with a softly elongated outline."
    elif jaw_to_cheek < 0.78 and upper_to_cheek > 1.02:
        face_shape = "Heart"
        shape_description = "The upper face reads wider than the jaw, with a narrower lower outline."
    elif jaw_to_cheek < 0.76:
        face_shape = "Diamond"
        shape_description = "The cheek area reads widest, with a narrower forehead and jaw outline."
    elif length_width_ratio <= 1.08 and jaw_to_cheek >= 0.88:
        face_shape = "Square"
        shape_description = "The face reads broad with a comparatively defined, similarly wide jaw."
    elif length_width_ratio <= 1.08:
        face_shape = "Round"
        shape_description = "The face reads close in length and width, with a softly tapered jaw."
    else:
        face_shape = "Oval"
        shape_description = "The face reads longer than it is wide, with a gently tapered jaw and balanced proportions."

    # Landmark-derived style estimates for the feature guide.
    brow_span = (distance(70, 107) + distance(300, 336)) / 2
    brow_arch = (
        abs((points[70][1] + points[107][1]) / 2 - points[105][1])
        + abs((points[300][1] + points[336][1]) / 2 - points[334][1])
    ) / 2
    arch_ratio = brow_arch / max(brow_span, 1.0)
    brows = "Defined arch" if arch_ratio >= 0.13 else "Soft angled" if arch_ratio >= 0.075 else "Mostly straight"

    eye_width = (distance(33, 133) + distance(362, 263)) / 2
    eye_opening = (distance(159, 145) + distance(386, 374)) / 2
    eye_aspect = eye_width / max(eye_opening, 1.0)
    eyes = "Almond" if eye_aspect >= 2.45 else "Rounded almond" if eye_aspect >= 1.9 else "Round"

    cheek_jaw_ratio = face_width / max(jaw_width, 1.0)
    cheekbones = "Prominent" if cheek_jaw_ratio >= 1.28 else "Balanced" if cheek_jaw_ratio >= 1.12 else "Softly defined"

    nose_width = distance(98, 327)
    nose_length = distance(168, 2)
    nose_width_ratio = nose_width / face_width
    nose_length_ratio = nose_length / face_length
    nose_width_label = "Narrow" if nose_width_ratio < 0.19 else "Proportional" if nose_width_ratio < 0.26 else "Broad"
    nose_length_label = "Short" if nose_length_ratio < 0.25 else "Medium length" if nose_length_ratio < 0.34 else "Long"

    lip_width = distance(61, 291)
    lip_height = distance(0, 17)
    lip_width_ratio = lip_width / face_width
    lip_fullness_ratio = lip_height / max(lip_width, 1.0)
    lip_width_label = "Narrow" if lip_width_ratio < 0.36 else "Balanced width" if lip_width_ratio < 0.46 else "Broad"
    lip_fullness_label = "Fuller outline" if lip_fullness_ratio >= 0.27 else "Medium outline" if lip_fullness_ratio >= 0.18 else "Defined outline"
    feature_profile = {
        "face_shape": face_shape,
        "brows": brows,
        "eyes": eyes,
        "cheekbones": cheekbones,
        "nose": f"{nose_width_label} width · {nose_length_label.lower()}",
        "lips": f"{lip_width_label} · {lip_fullness_label.lower()}",
    }
    # CIELAB Optical Skin Undertone
    skin_indices = [117, 346, 50, 280, 10]
    samples = []
    for s_idx in skin_indices:
        px, py = int(points[s_idx][0]), int(points[s_idx][1])
        px = max(3, min(width - 4, px))
        py = max(3, min(height - 4, py))
        patch = rgb[py - 2 : py + 3, px - 2 : px + 3]
        samples.append(np.median(patch, axis=(0, 1)))
    median_rgb = np.median(samples, axis=0).astype(int)
    hex_color = "#{:02x}{:02x}{:02x}".format(*median_rgb)
    rgb_1px = np.uint8([[median_rgb]])
    lab_1px = cv2.cvtColor(rgb_1px, cv2.COLOR_RGB2LAB)[0][0]
    l_val = float(lab_1px[0])
    a_star = float(lab_1px[1]) - 128.0
    b_star = float(lab_1px[2]) - 128.0
    if b_star > 5.0:
        undertone = "Warm (Golden / Peachy)"
        undertone_note = "Your complexion reflects golden or honey warmth that harmonizes with warm earth tones, peach, and coral accents."
        if l_val > 155:
            season = "Spring Radiance"
            palette = [
                ("#E07A5F", "Terracotta Coral"),
                ("#F4A261", "Warm Peach"),
                ("#2A9D8F", "Warm Turquoise"),
                ("#E9C46A", "Soft Marigold"),
                ("#F7EDE2", "Ivory Cream"),
            ]
        else:
            season = "Autumn Glow"
            palette = [
                ("#9E2A2B", "Deep Brick"),
                ("#DDA15E", "Caramel Amber"),
                ("#606C38", "Olive Moss"),
                ("#BC6C25", "Spiced Ochre"),
                ("#540B0E", "Rich Burgundy"),
            ]
    elif b_star < -1.0 or (a_star > 10.0 and b_star < 3.0):
        undertone = "Cool (Rosy / Muted Pink)"
        undertone_note = "Your complexion shows gentle rosy or cool blue-pink undertones that pair elegantly with soft berries, silver, and jewel hues."
        if l_val > 155:
            season = "Summer Elegance"
            palette = [
                ("#D4A5B8", "Soft Rose"),
                ("#A3B18A", "Sage Green"),
                ("#9381FF", "Soft Lavender"),
                ("#B8C0FF", "Powder Blue"),
                ("#E8D7F1", "Icy Lilac"),
            ]
        else:
            season = "Winter Contrast"
            palette = [
                ("#7209B7", "Royal Violet"),
                ("#1F4E5B", "Deep Emerald"),
                ("#9A031E", "True Crimson"),
                ("#1D3557", "Classic Navy"),
                ("#F8F9FA", "Pure White"),
            ]
    else:
        undertone = "Neutral (Harmonious Equilibrium)"
        undertone_note = "Your complexion displays a balanced equilibrium between warm and cool tones, offering versatility across both palettes."
        season = "Soft Classical"
        palette = [
            ("#C47AC0", "Dusty Orchid"),
            ("#E0A96D", "Warm Sand"),
            ("#778DA9", "Slate Blue"),
            ("#A5A58D", "Soft Khaki"),
            ("#E7C6FF", "Rose Quartz"),
        ]
    geometry = {
        "face_length": face_length,
        "face_width": face_width,
        "ratio": length_width_ratio,
        "face_shape": face_shape,
        "shape_description": shape_description,
        "features": feature_profile,
        "upper_pct": upper_pct,
        "middle_pct": middle_pct,
        "lower_pct": lower_pct,
        "total_thirds": total_thirds,
    }
    color_data = {
        "hex": hex_color,
        "rgb": median_rgb.tolist(),
        "undertone": undertone,
        "undertone_note": undertone_note,
        "season": season,
        "palette": palette,
    }
    return (img_copy, mesh_image, geometry, color_data), None

def add_to_favorites(category, title, detail):
    fav_item = {"category": category, "title": title, "detail": detail}
    if fav_item not in st.session_state.favorites:
        st.session_state.favorites.append(fav_item)
        st.toast(f"Saved to your Capsule: {title} 💖", icon="✨")
    else:
        st.toast("Already in your Capsule!", icon="🌸")

# ==============================================================================
# 5. HAUTE ATELIER SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; padding: 12px 0 16px 0;">
            <div class="atelier-monogram">G</div>
            <div style="display:flex; flex-direction:column;">
                <span style="font-family:'Playfair Display', serif; font-size:18px; font-weight:700; color:var(--color-primary); line-height:1.1;">Glow AI</span>
                <span style="font-size:10px; color:var(--color-secondary); letter-spacing:0.18em; text-transform:uppercase; font-weight:600;">Haute Atelier</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    studio_options = [
        "Overview / Studio",
        "Face & Color Analysis",
        "Virtual Try-On & Makeup",
        "Hairstyle & Cuts",
        "Eyewear Studio",
        "Skincare & Product Advisor",
        "AI Beauty Stylist",
        f"Saved Looks & Capsule ({len(st.session_state.favorites)})",
    ]
    option_clean_names = [s.split("(")[0].strip() for s in studio_options]
    active_clean = st.session_state.active_studio_name.split("(")[0].strip()
    try:
        active_index = option_clean_names.index(active_clean)
    except ValueError:
        active_index = 0
    selected_studio = st.radio(
        "ATELIER STUDIOS",
        studio_options,
        index=active_index,
        key="atelier_sidebar_nav",
        label_visibility="collapsed",
    )
    st.session_state.active_studio_name = selected_studio.split("(")[0].strip()
    st.markdown("<hr style='border-color: #d7c1c3; margin: 16px 0;'>", unsafe_allow_html=True)
    if st.session_state.analyzed:
        st.markdown(
            f"""
            <div style="background: #ffffff; border-radius: 16px; padding: 14px; border: 1px solid #d7c1c3; box-shadow: var(--shadow-ambient); margin-bottom: 12px;">
                <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--color-primary); font-weight: 700;">Scanned Visage Profile</div>
                <div style="font-size: 15px; font-weight: 700; color: #201a1c; margin-top: 3px;">{st.session_state.geometry.get('face_shape', 'Analyzed')}</div>
                <div style="font-size: 12px; color: #524344; margin-top: 2px;">Proportions: {st.session_state.geometry.get('ratio', 0.0):.2f} : 1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Scan New Portrait", key="sidebar_reset_btn", use_container_width=True):
            reset_all_visage_state()
            st.rerun()
    st.markdown(
        """
        <div style="border-radius: 18px; padding: 14px; background: linear-gradient(135deg, #fdf1f3 0%, #f7ebed 50%, #ffdadc 100%); border: 1px solid #d7c1c3; margin-top: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--color-primary); font-weight:700;">VIP Privilège</span>
                <span style="font-size:14px; color:var(--color-secondary);">✦</span>
            </div>
            <div style="font-size: 12px; font-weight: 700; color: #201a1c; margin-top: 2px;">Glow Luxe Member</div>
            <div style="font-size: 11px; color: #524344;">Local In-Memory Studio Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size: 10.5px; color: #857374; margin-top: 16px; line-height: 1.4;">
            🔒 <strong>100% Client-Side Privacy:</strong> All facial landmark mapping and color analysis run locally in-memory during this session.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 6. TOP NAVIGATION BAR (STICKY ATELIER HEADER)
# ==============================================================================
st.markdown(
    f"""
    <div class="stitch-topbar">
        <div style="display:flex; align-items:center; gap:12px;">
            <span class="stitch-pill stitch-pill-secondary">✦ Atelier Session</span>
            <span style="font-size: 13px; color: var(--color-on-surface-variant);">Active Studio: <strong>{st.session_state.active_studio_name}</strong></span>
        </div>
        <div style="display:flex; align-items:center; gap:14px;">
            <span class="stitch-pill">Capsule: {len(st.session_state.favorites)} Curations</span>
            <span style="font-size: 13px; font-weight:600; color:var(--color-primary);">Bonjour, Atelier Guest</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 7. MAIN VIEW ROUTER
# ==============================================================================
if "Overview / Studio" in st.session_state.active_studio_name:
    st.markdown(
        f"""
        <div class="stitch-hero">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 20px;">
                <div style="max-width: 650px;">
                    <div class="stitch-pill stitch-pill-secondary" style="margin-bottom: 10px;">✦ GOOGLE STITCH · HAUTE ATELIER</div>
                    <h1 style="font-size: 38px; margin: 4px 0 10px 0; font-weight: 600; line-height: 1.15;">
                        AI Skincare & Facial Architecture Studio
                    </h1>
                    <p style="color: var(--color-on-surface-variant); font-size: 15px; line-height: 1.55; margin: 0 0 14px 0;">
                        Where clinical facial geometry meets bespoke aesthetic intelligence. Upload your portrait to map 468 facial mesh coordinates, evaluate structural thirds balance, and discover custom circadian rituals calibrated for your visage.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span class="stitch-pill" style="font-size: 10px;">468-Point Mesh</span>
                        <span class="stitch-pill" style="font-size: 10px;">CIELAB Undertone</span>
                        <span class="stitch-pill" style="font-size: 10px;">Circadian AM/PM</span>
                        <span class="stitch-pill" style="font-size: 10px;">100% In-Memory Privacy</span>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: -12px; background: rgba(255,255,255,0.85); padding: 12px 18px; border-radius: 24px; border: 1px solid #d7c1c3; box-shadow: 0 4px 16px rgba(196,124,133,0.08);">
                    <div style="display: flex; margin-right: 14px;">
                        <img src="{DIVERSE_VISAGES[0]['img_file']}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2.5px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Camille"/>
                        <img src="{DIVERSE_VISAGES[1]['img_file']}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2.5px solid white; margin-left: -14px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Model 1"/>
                        <img src="{DIVERSE_VISAGES[2]['img_file']}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2.5px solid white; margin-left: -14px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" alt="Model 2"/>
                    </div>
                    <div>
                        <div style="font-size: 13px; font-weight: 700; color: #201a1c;">Atelier Muses Gallery</div>
                        <div style="font-size: 11px; color: var(--color-primary); font-weight: 600;">5 Visages Calibrated</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; margin-bottom: 12px;">
            <div>
                <span class="stitch-pill stitch-pill-secondary">EDITORIAL VISAGE SHOWCASE</span>
                <h2 style="font-size: 24px; margin: 4px 0 0 0;">Beauty Inspiration & Visage Gallery</h2>
                <p style="font-size: 13px; color: var(--color-on-surface-variant); margin: 2px 0 0 0;">
                    Explore curated adult female visages showcasing diverse hairstyles, cosmetic placement, and skin undertones:
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gallery_cols = st.columns(5, gap="medium")
    for idx, vis in enumerate(DIVERSE_VISAGES):
        with gallery_cols[idx]:
            st.markdown(
                f"""
                <div class="visage-gallery-card">
                    <div style="position: relative; width: 100%; aspect-ratio: 4/5; overflow: hidden; background: #f7ebed; border-radius: 20px 20px 0 0;">
                        <img src="{vis['img_file']}" style="display:block; width:100%; height:100%; object-fit:cover;" alt="{vis['look']}" onerror="this.onerror=null;this.src='{_local_img_b64('image_placeholder.png')}'"/>
                        <div style="position: absolute; inset: 0; background: linear-gradient(to top, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.1) 40%, transparent 100%);"></div>
                        <span style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.92); color: #884a52; font-size: 9.5px; font-weight: 700; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase;">{vis['tag']}</span>
                        <div style="position: absolute; bottom: 10px; left: 12px; right: 12px; color: white;">
                            <div style="font-family: 'Playfair Display', serif; font-size: 14px; font-weight: 700;">{vis['look']}</div>
                            <div style="font-size: 10.5px; opacity: 0.9;">{vis['demographic']}</div>
                        </div>
                    </div>
                    <div style="padding: 12px; display: flex; flex-direction: column; gap: 6px;">
                        <div style="font-size: 11px; color: var(--color-on-surface);"><strong>Hair:</strong> {vis['hairstyle']}</div>
                        <div style="font-size: 11px; color: var(--color-on-surface);"><strong>Makeup:</strong> {vis['makeup']}</div>
                        <div style="font-size: 10.5px; color: var(--color-primary); font-weight: 600;">Tone: {vis['undertone_hint']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<hr style='border-color: #d7c1c3; margin: 28px 0;'>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <span class="stitch-pill stitch-pill-secondary">LIVE VISAGE SCANNER</span>
            <h2 style="font-size: 24px; margin: 4px 0 2px 0;">Portrait Upload & Inspection</h2>
            <p style="font-size: 13px; color: var(--color-on-surface-variant); margin: 0;">
                Provide your front-facing portrait to inspect facial geometry and generate your personalized atelier blueprint.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not st.session_state.analyzed:
        tab_up, tab_cam, tab_demo = st.tabs(
            [
                "📁 Upload Portrait Photo",
                "📸 Capture Live Selfie",
                "✨ Try Studio Sample Visage",
            ]
        )
        with tab_up:
            uploaded_file = st.file_uploader(
                "Upload front-facing photo (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                key=f"portrait_uploader_{st.session_state.uploader_key}",
                help="Upload a clear, front-facing portrait to preview immediately.",
            )
            if uploaded_file is not None:
                if (
                    st.session_state.staged_image is None
                    or st.session_state.last_uploaded_name != uploaded_file.name
                ):
                    try:
                        uploaded_file.seek(0)
                        img_loaded = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
                        img_loaded.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                        st.session_state.staged_image = img_loaded.copy()
                        st.session_state.last_uploaded_name = uploaded_file.name
                        st.session_state.staged_image_source = f"Uploaded Photo: {uploaded_file.name}"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load uploaded image: {exc}")
        with tab_cam:
            camera_pic = st.camera_input(
                "Capture live selfie portrait",
                key=f"camera_portrait_{st.session_state.uploader_key}",
            )
            if camera_pic is not None:
                try:
                    camera_bytes = camera_pic.getvalue()
                    capture_digest = hashlib.sha256(camera_bytes).hexdigest()
                    if capture_digest != st.session_state.last_camera_capture_digest:
                        camera_pic.seek(0)
                        img_cam = ImageOps.exif_transpose(Image.open(camera_pic)).convert("RGB")
                        img_cam.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                        st.session_state.staged_image = img_cam.copy()
                        st.session_state.staged_image_source = "Live Camera Capture"
                        st.session_state.last_camera_capture_digest = capture_digest
                        st.session_state.camera_analysis_attempted = False
                except Exception as exc:
                    st.error(f"Could not load camera photo: {exc}")
            elif st.session_state.staged_image_source == "Live Camera Capture":
                # Streamlit's camera clear/retake action clears both the widget and staged portrait.
                reset_all_visage_state()
        with tab_demo:
            st.markdown(
                """
                <div style="padding: 4px 0 12px 0; font-size: 13.5px; color: var(--color-on-surface-variant);">
                    Load our synthetic studio portrait to test the landmark detection and color engine without uploading a personal photo:
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🌟 Load Studio Demo Portrait", key="btn_load_synth_portrait"):
                demo_visage = generate_sample_portrait()
                st.session_state.staged_image = demo_visage
                st.session_state.staged_image_source = "Studio Demo Visage"
                st.rerun()
                
        if st.session_state.staged_image is not None:
            st.markdown(
                """
                <div style="margin: 22px 0 10px 0;">
                    <span class="stitch-pill stitch-pill-active">✦ PORTRAIT INSPECTION & PREVIEW</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prev_col1, prev_col2 = st.columns([1, 1.3], gap="large")
            with prev_col1:
                _show_image_inline(
                    st.session_state.staged_image,
                    f"Staged Portrait: {st.session_state.staged_image_source}",
                )
            with prev_col2:
                w_px, h_px = st.session_state.staged_image.size
                st.markdown(
                    f"""
                    <div class="stitch-card">
                        <span class="stitch-pill stitch-pill-secondary">READY FOR ANALYSIS</span>
                        <h3 style="margin: 8px 0 4px 0; font-size: 24px;">Visage Loaded Successfully</h3>
                        <div style="font-size: 13px; color: var(--color-on-surface-variant); margin-bottom: 12px;">
                            <strong>Source:</strong> {st.session_state.staged_image_source}<br>
                            <strong>Dimensions:</strong> {w_px} × {h_px} px · 24-bit RGB
                        </div>
                        <p style="font-size: 13.5px; color: var(--color-on-surface-variant); line-height: 1.5; margin-bottom: 16px;">
                            Our local MediaPipe FaceMesh engine will extract 468 landmark points to calculate your length-to-width ratio, measure vertical thirds distribution, and sample optical undertones in CIELAB space.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                auto_camera_analysis = (
                    st.session_state.staged_image_source == "Live Camera Capture"
                    and not st.session_state.camera_analysis_attempted
                )
                if auto_camera_analysis:
                    # Analyze this captured image once; a later Streamlit rerun must not restart it.
                    st.session_state.camera_analysis_attempted = True
                run_manual_analysis = st.button(
                    "✨ Execute Facial Architecture Analysis",
                    type="primary",
                    use_container_width=True,
                    key="btn_run_analysis_action",
                )
                if auto_camera_analysis or run_manual_analysis:
                    with st.spinner("Processing facial geometry & optical color metrics..."):
                        res_data, err_msg = analyze_portrait_landmarks(st.session_state.staged_image)
                        if err_msg:
                            st.error(err_msg)
                        else:
                            orig_p, mesh_p, geom_p, col_p = res_data
                            st.session_state.original_image = orig_p
                            st.session_state.mesh_image = mesh_p
                            st.session_state.geometry = geom_p
                            st.session_state.color_data = col_p
                            st.session_state.analyzed = True
                            st.toast("Facial architecture analysis complete! ✨")
                            st.rerun()
                if st.button("🗑️ Remove Photo / Choose Another", use_container_width=True, key="btn_remove_staged_photo"):
                    reset_all_visage_state()
                    st.rerun()
        else:
            st.markdown(
                """
                <div style="margin: 22px 0 10px 0;">
                    <span class="stitch-pill stitch-pill-active">✦ PORTRAIT INSPECTION & PREVIEW</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prev_col1, prev_col2 = st.columns([1, 1.3], gap="large")
            with prev_col1:
                st.markdown(
                    """
                    <figure style="margin:0;width:100%;padding:12px;border:1px solid #d7c1c3;border-radius:20px;background:#fff;box-shadow:0 8px 24px rgba(136,74,82,0.08);box-sizing:border-box;">
                        <div style="width:100%;aspect-ratio:4/5;min-height:300px;max-height:640px;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:12px;background:linear-gradient(155deg,#fff9fa 0%,#f8e9ec 100%);">
                            <svg viewBox="0 0 280 350" role="img" aria-label="Minimal front-facing face outline placeholder" style="width:78%;height:78%;max-width:290px;max-height:360px;">
                                <path d="M140 38 C91 38 60 75 60 132 C60 198 91 249 140 267 C189 249 220 198 220 132 C220 75 189 38 140 38Z" fill="rgba(255,255,255,0.34)" stroke="#c47c85" stroke-width="3" stroke-linecap="round"/>
                                <path d="M60 125 C48 115 45 135 53 153 M220 125 C232 115 235 135 227 153" fill="none" stroke="#d7a3aa" stroke-width="3" stroke-linecap="round"/>
                                <path d="M106 132 Q121 124 132 132 M148 132 Q159 124 174 132 M140 135 C136 158 132 171 140 175 M123 199 Q140 207 157 199" fill="none" stroke="#d7a3aa" stroke-width="2.5" stroke-linecap="round"/>
                                <path d="M112 267 L108 292 Q140 309 172 292 L168 267 M108 292 Q81 301 68 323 M172 292 Q199 301 212 323" fill="none" stroke="#c47c85" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M76 322 Q140 302 204 322" fill="none" stroke="#e2bdc2" stroke-width="3" stroke-linecap="round"/>
                            </svg>
                        </div>
                    </figure>
                    """,
                    unsafe_allow_html=True,
                )
            with prev_col2:
                st.markdown(
                    """
                    <div class="stitch-card" style="min-height:300px;display:flex;flex-direction:column;justify-content:center;">
                        <span class="stitch-pill stitch-pill-secondary">YOUR PORTRAIT</span>
                        <h3 style="margin:12px 0 8px 0;font-size:24px;">Your portrait preview will appear here</h3>
                        <p style="font-size:13.5px;color:var(--color-on-surface-variant);line-height:1.55;margin:0;">
                            Upload a clear, front-facing portrait with your full face visible and even lighting. You can also capture a live selfie or try the studio sample above.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div style="margin: 10px 0 14px 0;">
                <span class="stitch-pill stitch-pill-active">✦ ACTIVE SCANNED VISAGE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        geom = st.session_state.geometry
        color = st.session_state.color_data
        col_post1, col_post2 = st.columns([1.1, 1.2], gap="large")
        with col_post1:
            view_toggle = st.radio(
                "Visage Layer Display",
                ["Natural Portrait", "AI Landmark Mesh", "Side-by-Side View"],
                horizontal=True,
                key="ov_post_toggle",
            )
            if view_toggle == "Natural Portrait":
                _show_image_inline(st.session_state.original_image, caption="Natural Portrait")
            elif view_toggle == "AI Landmark Mesh":
                _show_image_inline(st.session_state.mesh_image, caption="MediaPipe 468-Point Mesh Wireframe")
            else:
                s_a, s_b = st.columns(2)
                with s_a:
                    _show_image_inline(st.session_state.original_image, caption="Natural")
                with s_b:
                    _show_image_inline(st.session_state.mesh_image, caption="AI Mesh")
        with col_post2:
            st.markdown(
                f"""
                <div class="stitch-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="stitch-pill stitch-pill-secondary">ANALYSIS RESULTS</span>
                        <span class="stitch-pill stitch-pill-active">Ratio: {geom['ratio']:.2f} : 1</span>
                    </div>
                    <h3 style="font-size: 24px; margin: 8px 0 2px 0;">{geom['face_shape']} Architecture</h3>
                    <p style="font-size: 13.5px; color: var(--color-on-surface-variant); line-height: 1.5; margin: 6px 0 14px 0;">
                        {geom['shape_description']}
                    </p>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;">
                        <div style="background:#fdf1f3; border:1px solid #d7c1c3; border-radius:14px; padding:12px;">
                            <span style="font-size:10px; font-weight:700; color:var(--color-primary); text-transform:uppercase;">Complexion Tone</span>
                            <div style="font-size:13.5px; font-weight:700; color:#201a1c; margin-top:2px;">{color['undertone'].split('(')[0].strip()}</div>
                        </div>
                        <div style="background:#fdf1f3; border:1px solid #d7c1c3; border-radius:14px; padding:12px;">
                            <span style="font-size:10px; font-weight:700; color:var(--color-primary); text-transform:uppercase;">Seasonal Palette</span>
                            <div style="font-size:13.5px; font-weight:700; color:#201a1c; margin-top:2px;">{color['season']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_feature_profile(geom.get("features", {}))
            col_btn_a, col_btn_b = st.columns(2)
            with col_btn_a:
                st.button(
                    "📐 Open Face Analysis",
                    use_container_width=True,
                    key="btn_goto_face",
                    on_click=navigate_to_studio,
                    args=("Face & Color Analysis",),
                )
            with col_btn_b:
                st.button(
                    "🧴 Open Skincare Advisor",
                    use_container_width=True,
                    key="btn_goto_skin",
                    on_click=navigate_to_studio,
                    args=("Skincare & Product Advisor",),
                )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Upload / Scan Another Photo", type="primary", use_container_width=True, key="btn_reset_from_ov_complete"):
                reset_all_visage_state()
                st.rerun()

elif "Face & Color Analysis" in st.session_state.active_studio_name:
    if not st.session_state.analyzed and st.session_state.staged_image is not None:
        try:
            with st.spinner("Analyzing your saved portrait with MediaPipe FaceMesh..."):
                res_data, err_msg = analyze_portrait_landmarks(st.session_state.staged_image)
            if err_msg:
                st.error(err_msg)
            else:
                orig_p, mesh_p, geom_p, col_p = res_data
                st.session_state.original_image = orig_p
                st.session_state.mesh_image = mesh_p
                st.session_state.geometry = geom_p
                st.session_state.color_data = col_p
                st.session_state.analyzed = True
        except Exception as exc:
            st.error(f"Portrait analysis could not complete: {exc}")
    if (
        not st.session_state.analyzed
        and st.session_state.original_image is not None
        and st.session_state.mesh_image is not None
        and st.session_state.geometry
        and st.session_state.color_data
    ):
        # Keep complete saved results usable if a session's analyzed flag was lost.
        st.session_state.analyzed = True
    if not st.session_state.analyzed:
        if st.session_state.staged_image is None and st.session_state.original_image is None:
            st.info("Please scan or upload a portrait in 'Overview / Studio' first to enable real-time face and color analysis.")
        else:
            st.error("Your portrait is available, but its analysis could not be completed. Return to Overview / Studio and try the analysis again.")
    else:
        geom = st.session_state.geometry
        color = st.session_state.color_data
        st.markdown(
            f"""
            <div class="stitch-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <span class="stitch-pill stitch-pill-secondary">LIVE VISAGE GEOMETRY</span>
                        <h2 style="font-size:26px; margin:4px 0 0 0;">{geom['face_shape']} Proportions</h2>
                    </div>
                    <span class="stitch-pill stitch-pill-active">Ratio: {geom['ratio']:.2f} : 1</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_vis, col_data = st.columns([1.1, 1], gap="large")
        with col_vis:
            mode = st.radio(
                "Visual Inspection Layer",
                ["Side-by-Side Comparison", "AI Mesh Tessellation", "Natural Portrait"],
                horizontal=True,
            )
            if mode == "Side-by-Side Comparison":
                sub1, sub2 = st.columns(2)
                with sub1:
                    _show_image_inline(st.session_state.original_image, caption="Natural Portrait")
                with sub2:
                    _show_image_inline(st.session_state.mesh_image, caption="MediaPipe 468 Tessellation")
            elif mode == "AI Mesh Tessellation":
                _show_image_inline(st.session_state.mesh_image, caption="MediaPipe 468-Point Mesh")
            else:
                _show_image_inline(st.session_state.original_image, caption="Natural Portrait")
        with col_data:
            if geom["total_thirds"] > 0:
                labels = ["Upper Segment", "Middle Segment", "Lower Segment"]
                vals = [geom["upper_pct"], geom["middle_pct"], geom["lower_pct"]]
                fig = go.Figure(
                    go.Bar(
                        x=vals,
                        y=labels,
                        orientation="h",
                        marker=dict(
                            color=["#ffd9dc", "#fdc7cb", "#884a52"],
                            line=dict(color="#ffffff", width=1.5),
                        ),
                        text=[f"{v:.1f}%" for v in vals],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text="Vertical Thirds Distribution",
                        font=dict(family="Playfair Display", size=16, color="#201a1c"),
                    ),
                    xaxis=dict(range=[0, 100], gridcolor="#f7ebed", tickfont=dict(color="#524344")),
                    yaxis=dict(autorange="reversed", tickfont=dict(color="#524344")),
                    height=220,
                    margin=dict(l=10, r=10, t=35, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#201a1c", size=12),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"""
                <div class="stitch-card" style="margin-top: 14px;">
                    <span class="stitch-pill stitch-pill-tertiary">CIELAB UNDERTONE</span>
                    <h3 style="font-size: 20px; margin: 6px 0 4px 0;">{color['undertone']}</h3>
                    <p style="font-size: 13px; color: var(--color-on-surface-variant); line-height: 1.45; margin-bottom: 12px;">
                        {color['undertone_note']}
                    </p>
                    <div style="display:flex; align-items:center; gap:12px; padding:10px 14px; background:#fdf1f3; border-radius:14px; border:1px solid #d7c1c3;">
                        <div style="width:32px; height:32px; border-radius:50%; background-color:{color['hex']}; border:2px solid white; box-shadow:0 2px 6px rgba(0,0,0,0.15);"></div>
                        <div>
                            <span style="font-size:11px; font-weight:700; color:var(--color-primary); text-transform:uppercase;">Extracted Median Skin Pigment</span>
                            <div style="font-size:13px; font-weight:600; color:#201a1c;">{color['hex'].upper()} · RGB {color['rgb']}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _render_feature_profile(geom.get("features", {}))
        st.markdown(f"### 🌸 Seasonal Palette: **{color['season']}**")
        pal_cols = st.columns(5)
        for idx, (hex_c, name) in enumerate(color["palette"]):
            with pal_cols[idx]:
                st.markdown(
                    f"""
                    <div class="stitch-card" style="text-align: center; padding: 14px 8px;">
                        <div style="width: 44px; height: 44px; border-radius: 50%; background: {hex_c}; margin: 0 auto 8px auto; border: 2px solid white; box-shadow: 0 4px 10px rgba(0,0,0,0.12);"></div>
                        <div style="font-size: 12px; font-weight: 700; color: #201a1c;">{name}</div>
                        <div style="font-size: 10.5px; color: #857374; font-family: monospace;">{hex_c}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif "Virtual Try-On & Makeup" in st.session_state.active_studio_name:
    shape = st.session_state.geometry.get("face_shape", "Oval / Balanced")
    st.markdown(
        f"""
        <div class="stitch-hero">
            <span class="stitch-pill stitch-pill-secondary">VIRTUAL COSMETIC ATELIER</span>
            <h1 style="font-size:32px;margin:6px 0 8px 0;">Virtual Makeup Try-On</h1>
            <p style="color:var(--color-on-surface-variant);font-size:14.5px;margin:0;">
                Upload a front-facing portrait and preview feature-aligned color with local FaceMesh processing. Your image stays in this session.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_makeup_file = st.file_uploader(
        "Upload a front-facing portrait",
        type=["jpg", "jpeg", "png"],
        key=f"virtual_makeup_file_{st.session_state.virtual_makeup_upload_key}",
        help="Use a clear, well-lit photo with the full face visible.",
    )
    if uploaded_makeup_file is not None:
        try:
            uploaded_makeup_file.seek(0)
            makeup_image = ImageOps.exif_transpose(Image.open(uploaded_makeup_file)).convert("RGB")
            makeup_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            st.session_state.virtual_makeup_image = makeup_image.copy()
            st.session_state.virtual_makeup_source = uploaded_makeup_file.name
        except Exception:
            st.session_state.virtual_makeup_image = None
            st.error("This image could not be opened. Please upload a valid JPG or PNG portrait.")

    makeup_image = st.session_state.virtual_makeup_image
    if makeup_image is None:
        st.info("Upload a front-facing portrait to start a real makeup preview.")
    else:
        lipstick_shades = {
            "Rosewood Nude": "#A94F68",
            "Petal Pink": "#C86B7D",
            "Classic Red": "#A72D3D",
            "Berry Wine": "#74364F",
            "Coral Bloom": "#D66B59",
        }
        blush_shades = {
            "Peach Bloom": "#E98979",
            "Petal Pink": "#D96F88",
            "Rose Mauve": "#B86479",
            "Warm Terracotta": "#C86D52",
        }
        eyeshadow_shades = {
            "Soft Rose": "#B97482",
            "Champagne": "#C99C72",
            "Warm Taupe": "#80665E",
            "Plum Smoke": "#69475E",
            "Bronze Glow": "#A66B3D",
        }
        controls_a, controls_b, controls_c = st.columns(3, gap="medium")
        with controls_a:
            lipstick_name = st.selectbox("Lipstick shade", list(lipstick_shades), key="makeup_lipstick")
            blush_name = st.selectbox("Blush shade", list(blush_shades), key="makeup_blush")
        with controls_b:
            blush_placement = st.selectbox(
                "Blush placement",
                ["Soft apples", "Lifted sweep", "Soft drape"],
                key="makeup_blush_placement",
            )
            eyeshadow_name = st.selectbox("Eyeshadow shade", list(eyeshadow_shades), key="makeup_eyeshadow")
        with controls_c:
            eyeliner_style = st.selectbox(
                "Eyeliner style",
                ["None", "Soft definition", "Classic wing"],
                key="makeup_eyeliner",
            )
            intensity = st.slider("Makeup intensity", 0, 100, value=65, key="makeup_intensity", format="%d%%")

        normalized_buffer = BytesIO()
        makeup_image.save(normalized_buffer, format="PNG", optimize=True)
        normalized_bytes = normalized_buffer.getvalue()
        try:
            with st.spinner("Mapping eyes, lips, and cheeks…"):
                makeup_points = _cached_makeup_face_points(normalized_bytes)
            if makeup_points is None:
                st.warning("No face was detected. Try a clear, front-facing portrait with even lighting.")
                _show_image_inline(makeup_image, f"Original portrait · {st.session_state.virtual_makeup_source}")
            else:
                makeup_preview = _virtual_makeup_preview(
                    makeup_image,
                    makeup_points,
                    lipstick_shades[lipstick_name],
                    blush_shades[blush_name],
                    blush_placement,
                    eyeshadow_shades[eyeshadow_name],
                    eyeliner_style,
                    intensity,
                )
                comparison = st.radio(
                    "Preview comparison",
                    ["Side-by-side", "Makeup preview", "Original photo"],
                    horizontal=True,
                    key="makeup_comparison",
                )
                if comparison == "Side-by-side":
                    before_col, after_col = st.columns(2, gap="medium")
                    with before_col:
                        _show_image_inline(makeup_image, f"Original · {st.session_state.virtual_makeup_source}")
                    with after_col:
                        _show_image_inline(makeup_preview, "Virtual makeup preview")
                elif comparison == "Makeup preview":
                    _show_image_inline(makeup_preview, "Virtual makeup preview")
                else:
                    _show_image_inline(makeup_image, f"Original · {st.session_state.virtual_makeup_source}")

                _save_col, _reset_col = st.columns(2, gap="medium")
                with _save_col:
                    if st.button("💖 Save Makeup Look", type="primary", use_container_width=True, key="btn_save_virtual_makeup"):
                        detail = (
                            f"Face shape: {shape}. Lipstick: {lipstick_name}; blush: {blush_name} ({blush_placement}); "
                            f"eyeshadow: {eyeshadow_name}; eyeliner: {eyeliner_style}; intensity: {intensity}%."
                        )
                        add_to_favorites("Virtual Makeup Look", f"{lipstick_name} Atelier Look", detail)
                with _reset_col:
                    st.button(
                        "↺ Reset Try-On",
                        use_container_width=True,
                        key="btn_reset_virtual_makeup",
                        on_click=_reset_virtual_makeup,
                    )
        except Exception as exc:
            st.error(f"The makeup preview could not be rendered: {exc}")

elif "Hairstyle & Cuts" in st.session_state.active_studio_name:
    st.markdown("""<div class="stitch-hero"><span class="stitch-pill stitch-pill-secondary">HAIR ARCHITECTURE & SILHOUETTE</span><h1 style="font-size:32px;margin:6px 0 8px;">Hairstyle Atelier</h1><p style="color:var(--color-on-surface-variant);font-size:14.5px;margin:0;">Face-shape-led hairstyle ideas with photographic references selected from the Glow AI portrait collection.</p></div>""",unsafe_allow_html=True)
    upload=st.file_uploader("Upload a front-facing portrait",type=["jpg","jpeg","png"],key=f"hair_file_{st.session_state.hair_upload_key}")
    if upload is not None:
        try:
            upload.seek(0); portrait=ImageOps.exif_transpose(Image.open(upload)).convert("RGB"); portrait.thumbnail((1000,1000),Image.Resampling.LANCZOS)
            st.session_state.hair_image=portrait.copy(); st.session_state.hair_source=upload.name
        except Exception:
            st.session_state.hair_image=None; st.error("This image could not be opened. Choose a valid JPG or PNG portrait.")
    if st.session_state.hair_image is None and st.session_state.original_image is not None:
        if st.button("Use the portrait already analyzed in Face & Color Analysis",key="use_existing_hair_portrait"):
            st.session_state.hair_image=st.session_state.original_image.copy(); st.session_state.hair_source="Face & Color Analysis portrait"
    portrait=st.session_state.hair_image
    if portrait is None:
        st.info("Upload a clear, front-facing portrait to receive three hairstyle recommendations.")
    else:
        b=BytesIO(); portrait.save(b,format="PNG")
        try:
            with st.spinner("Mapping facial proportions…"): points=_cached_style_face_points(b.getvalue())
        except Exception as exc:
            points=None; st.error(f"Portrait analysis could not complete: {exc}")
        if points is None:
            st.warning("No face was detected. Try a clear, front-facing portrait with both eyes and the full face visible.")
            _show_image_inline(portrait,f"Uploaded portrait · {st.session_state.hair_source}")
        else:
            shape=_estimate_style_face_shape(points); st.session_state.geometry["face_shape"]=shape
            _show_image_inline(portrait,f"Your portrait · {st.session_state.hair_source}")
            st.markdown(f"<div class='stitch-card' style='padding:16px 20px;margin:12px 0'><span class='stitch-pill stitch-pill-secondary'>DETECTED FACE SHAPE</span><h3 style='margin:8px 0 4px'>{html.escape(shape)}</h3><p style='margin:0;color:#766b6d'>{html.escape(_face_shape_explanation(shape))}</p></div>",unsafe_allow_html=True)
            recommendations=_style_recommendations(shape,"hair")
            st.markdown("### Three hairstyle recommendations")
            cards=st.columns(3,gap="medium")
            for i,(name,why,length,description,maintenance,asset_name,code) in enumerate(recommendations):
                with cards[i]:
                    _show_image_inline(_reference_photo(asset_name),f"Photo reference · {name}")
                    st.markdown(f"**0{i+1} · {name}**")
                    st.caption(why)
                    st.markdown(f"**Length:** {length}  \n**Style:** {description}  \n**Maintenance:** {maintenance}")
                    if st.button(f"Explore This Style · {i+1}",key=f"explore_hair_{i}",type="primary",use_container_width=True):
                        st.session_state.hair_selected_name=name
                        st.session_state.hair_selected_style=code
            selected=next((r for r in recommendations if r[0]==st.session_state.hair_selected_name),None)
            if selected:
                name,why,length,description,maintenance,asset_name,code=selected
                st.markdown("### Style details")
                detail_col,copy_col=st.columns([1,1.2],gap="large")
                with detail_col:
                    _show_image_inline(_reference_photo(asset_name),f"Hairstyle reference · {name}")
                with copy_col:
                    st.markdown(f"#### {name}")
                    st.markdown(f"**Why it suits {shape}:** {why}")
                    st.markdown(f"**Length:** {length}  \n**Description:** {description}  \n**Maintenance:** {maintenance}")
                    if st.button("💖 Save Hairstyle",key="save_hair_style",type="primary",use_container_width=True):
                        add_to_favorites("Hairstyle Recommendation",name,f"Recommended for {shape}. Length: {length}. {description} Maintenance: {maintenance}. {why}")

elif "Eyewear Studio" in st.session_state.active_studio_name:
    st.markdown("""<div class="stitch-hero"><span class="stitch-pill stitch-pill-secondary">OPTICAL & SUNGLASS ATELIER</span><h1 style="font-size:32px;margin:6px 0 8px;">Eyewear Studio</h1><p style="color:var(--color-on-surface-variant);font-size:14.5px;margin:0;">Personalized frame recommendations selected for your face shape, shown with frame-only product photos.</p></div>""",unsafe_allow_html=True)
    upload=st.file_uploader("Upload a front-facing portrait",type=["jpg","jpeg","png"],key=f"eyewear_file_{st.session_state.eyewear_upload_key}")
    if upload is not None:
        try:
            upload.seek(0); portrait=ImageOps.exif_transpose(Image.open(upload)).convert("RGB"); portrait.thumbnail((1000,1000),Image.Resampling.LANCZOS)
            st.session_state.eyewear_image=portrait.copy(); st.session_state.eyewear_source=upload.name
        except Exception:
            st.session_state.eyewear_image=None; st.error("This image could not be opened. Choose a valid JPG or PNG portrait.")
    if st.session_state.eyewear_image is None and st.session_state.original_image is not None:
        if st.button("Use the portrait already analyzed in Face & Color Analysis",key="use_existing_eyewear_portrait"):
            st.session_state.eyewear_image=st.session_state.original_image.copy(); st.session_state.eyewear_source="Face & Color Analysis portrait"
    portrait=st.session_state.eyewear_image
    if portrait is None:
        st.info("Upload a clear, front-facing portrait to receive personalized frame recommendations.")
    else:
        b=BytesIO(); portrait.save(b,format="PNG")
        try:
            with st.spinner("Mapping eye corners and nose bridge…"): points=_cached_style_face_points(b.getvalue())
        except Exception as exc:
            points=None; st.error(f"Portrait analysis could not complete: {exc}")
        if points is None:
            st.warning("No face was detected. Try a clear, front-facing portrait with both eyes visible.")
            _show_image_inline(portrait,f"Uploaded portrait · {st.session_state.eyewear_source}")
        else:
            shape=_estimate_style_face_shape(points); st.session_state.geometry["face_shape"]=shape
            st.markdown(f"<div class='stitch-card' style='padding:16px 20px;margin:12px 0'><span class='stitch-pill stitch-pill-secondary'>DETECTED FACE SHAPE</span><h3 style='margin:8px 0 4px'>{html.escape(shape)}</h3><p style='margin:0;color:#766b6d'>{html.escape(_face_shape_explanation(shape))}</p></div>",unsafe_allow_html=True)
            recommendations=_style_recommendations(shape,"eyewear")
            st.markdown("### Three personalized frame recommendations")
            cards=st.columns(3,gap="medium")
            for i,(name,why,design,palette,frame) in enumerate(recommendations):
                with cards[i]:
                    _show_image_inline(_eyewear_product_image(frame),f"Eyeglasses product photo · {name}")
                    st.markdown(f"**0{i+1} · {name}**"); st.caption(why)
                    st.markdown(f"**Frame shape:** {design}  \n**Colour suggestions:** {palette}")
                    if st.button(f"Explore This Frame · {i+1}",key=f"explore_eyewear_{i}",type="primary",use_container_width=True):
                        st.session_state.eyewear_selected_name=name
            selected_frame=next((item for item in recommendations if item[0]==st.session_state.eyewear_selected_name),None)
            if selected_frame:
                name,why,design,palette,frame=selected_frame
                about,styling_tip=_EYEWEAR_FRAME_INFO.get(frame,(
                    f"A thoughtfully proportioned {design.lower()} selected for your estimated face shape.",
                    "Keep the frame comfortable and balanced with your brow line and everyday accessories.",
                ))
                st.markdown("### Frame details")
                with st.container(border=True):
                    st.markdown(
                        f"<div class='stitch-card' style='padding:18px 20px;margin:0 0 12px'>"
                        f"<span class='stitch-pill stitch-pill-secondary'>PERSONALIZED FRAME PROFILE</span>"
                        f"<h3 style='margin:10px 0 8px'>{html.escape(name)}</h3>"
                        f"<p style='margin:0 0 12px;color:#766b6d'><strong>Why it may suit {html.escape(shape)}:</strong> {html.escape(why)}</p>"
                        f"<p style='margin:0 0 12px;color:#766b6d'><strong>About this frame:</strong> {html.escape(about)}</p>"
                        f"<p style='margin:0 0 12px;color:#766b6d'><strong>Frame shape:</strong> {html.escape(design)}</p>"
                        f"<p style='margin:0 0 12px;color:#766b6d'><strong>Colour suggestions:</strong> {html.escape(palette)}</p>"
                        f"<p style='margin:0;color:#766b6d'><strong>Styling tips:</strong> {html.escape(styling_tip)}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("💖 Save Frame",key="save_eyewear_frame",type="primary",use_container_width=True):
                        add_to_favorites(
                            "Eyewear Frame",
                            name,
                            f"Recommended for {shape}. Why: {why} About: {about} Frame shape: {design}. Suggested colours: {palette}. Styling tip: {styling_tip}.",
                        )

elif "Skincare & Product Advisor" in st.session_state.active_studio_name:
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:var(--color-on-surface-variant); font-weight:600;">
                AI BEAUTY STYLIST <span style="color:var(--color-outline);">›</span> <span style="color:var(--color-primary); font-weight:700;">SKINCARE & BESPOKE PRODUCT ADVISOR</span>
            </div>
            <span class="stitch-pill stitch-pill-secondary">✦ Adaptive Bio-Regimen</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 36px; margin: 0 0 6px 0; font-weight:600;">
                AI Skincare & <span style="font-family:'Playfair Display',serif; font-style:italic; color:var(--color-primary); font-weight:400;">Bespoke</span> Product Advisor
            </h1>
            <p style="color: var(--color-on-surface-variant); font-size: 14.5px; margin: 0; max-width: 820px; line-height: 1.5;">
                Personalized cosmetic formulations, circadian ritual choreography, and ingredient science calibrated for your dynamic skin profile.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    advisor_quotes = {
        "combination": "Welcome to your skincare studio. Today's seasonal formulation favors lightweight ceramide emulsions with targeted sebum-balancing in the T-zone.",
        "oily": "Pore balance calibration: Light humectant water gels and salicylic acid support follicular clarity without over-drying.",
        "dry": "Barrier repair priority: High-concentration physiological ceramides and multi-molecular hyaluronic acid restoring dermal hydration.",
        "normal": "Optimal equilibrium: Vitamin C photoprotection and breathable lipid fluids maintaining radiant daily balance.",
        "sensitive": "Hypoallergenic focus: Madecassoside calming milk and pure physical zinc shields soothing active reactivity.",
    }
    current_type = st.session_state.selected_skin_type
    current_quote = advisor_quotes.get(current_type, advisor_quotes["combination"])
    st.markdown(
        f"""
        <div class="stitch-card" style="padding: 24px; background: linear-gradient(135deg, #ffffff 0%, #fdf1f3 50%, #f7ebed 100%);">
            <div style="display: flex; flex-wrap: wrap; gap: 24px; align-items: center;">
                <div style="display: flex; gap: 16px; align-items: center; max-width: 480px;">
                    <div style="position: relative; width: 88px; height: 88px; border-radius: 20px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); shrink-0; background: white;">
                        <img src="{_IMG_DOCTOR}" style="display:block; width:100%; height:100%; object-fit:cover;" alt="AI Skincare Stylist" onerror="this.onerror=null;this.src='{_local_img_b64('image_placeholder.png')}'"/>
                        <span style="position: absolute; bottom: 4px; right: 4px; width: 10px; height: 10px; border-radius: 50%; background: #10b981; border: 2px solid white;" title="Available"></span>
                    </div>
                    <div>
                        <span class="stitch-pill stitch-pill-secondary" style="font-size: 9.5px; padding: 2px 8px;">AI Skincare & Beauty Stylist</span>
                        <h3 style="font-size: 19px; margin: 4px 0 2px 0;">Dr. Elise Moreau Persona</h3>
                        <p style="font-size: 12px; color: var(--color-on-surface-variant); margin: 0; line-height: 1.45; font-style: italic;">
                            "{current_quote}"
                        </p>
                    </div>
                </div>
                <div style="flex: 1; display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px;">
                    <div style="background: white; border-radius: 14px; padding: 12px; border: 1px solid #d7c1c3;">
                        <span style="font-size: 10px; text-transform: uppercase; color: var(--color-secondary); font-weight: 700;">Equilibrium</span>
                        <div style="font-size: 14px; font-weight: 700; color: #201a1c; margin-top: 2px;">{current_type.capitalize()}</div>
                        <span style="font-size: 10.5px; color: var(--color-primary);">Selected Skin Type</span>
                    </div>
                    <div style="background: white; border-radius: 14px; padding: 12px; border: 1px solid #d7c1c3;">
                        <span style="font-size: 10px; text-transform: uppercase; color: var(--color-secondary); font-weight: 700;">Barrier Target</span>
                        <div style="font-size: 14px; font-weight: 700; color: #201a1c; margin-top: 2px;">Lipid Balance</div>
                        <span style="font-size: 10.5px; color: #10b981;">AM & PM Care</span>
                    </div>
                    <div style="background: white; border-radius: 14px; padding: 12px; border: 1px solid #d7c1c3;">
                        <span style="font-size: 10px; text-transform: uppercase; color: var(--color-secondary); font-weight: 700;">Active Phase</span>
                        <div style="font-size: 14px; font-weight: 700; color: #201a1c; margin-top: 2px;">Circadian II</div>
                        <span style="font-size: 10.5px; color: var(--color-primary);">Day / Night Sync</span>
                    </div>
                    <div style="background: white; border-radius: 14px; padding: 12px; border: 1px solid #d7c1c3;">
                        <span style="font-size: 10px; text-transform: uppercase; color: var(--color-secondary); font-weight: 700;">Curated Formulations</span>
                        <div style="font-size: 14px; font-weight: 700; color: #201a1c; margin-top: 2px;">5 Products</div>
                        <span style="font-size: 10.5px; color: var(--color-primary);">Targeted Regimen</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="stitch-card">
            <span class="stitch-pill stitch-pill-secondary">ATELIER DIAGNOSTICS</span>
            <h3 style="font-size: 21px; margin: 4px 0 16px 0;">Interactive Profile & Dynamic Formulations</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    diag_col1, diag_col2, diag_col3 = st.columns([1.2, 1.2, 1], gap="medium")
    with diag_col1:
        st.markdown("**1. Select Skin Type**")
        st.caption("Calibrates product textures, actives, and hydration depth:")
        skin_choice = st.radio(
            "Skin Type Selection",
            ["Combination", "Oily", "Dry", "Normal", "Sensitive"],
            index=["combination", "oily", "dry", "normal", "sensitive"].index(st.session_state.selected_skin_type),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.selected_skin_type = skin_choice.lower()
    with diag_col2:
        st.markdown("**2. Primary Concerns (Select applicable)**")
        st.caption("Customizes targeted serums and barrier actives:")
        concern_options = ["Barrier Repair", "Dehydration", "Acne-prone", "Uneven Tone", "Fine Lines"]
        selected_c = st.multiselect(
            "Concerns Selection",
            concern_options,
            default=st.session_state.selected_concerns,
            label_visibility="collapsed",
        )
        st.session_state.selected_concerns = selected_c
    with diag_col3:
        st.markdown("**3. Tier / Price Point Preference**")
        st.caption("Filters budget categories for recommended products:")
        budget_choice = st.radio(
            "Budget Selection",
            ["Accessible ($16 – $28)", "Curated Mid ($32 – $58)", "Haute Luxury ($65 – $140)"],
            index=1,
            label_visibility="collapsed",
        )
        st.session_state.budget_tier = budget_choice
        
    st.markdown(
        """
        <div style="background: #fdf1f3; border: 1px solid var(--color-outline-variant); border-radius: 16px; padding: 14px 18px; margin: 16px 0 24px 0; font-size: 12px; color: var(--color-on-surface-variant); line-height: 1.5;">
            <strong>✦ Cosmetic Advisory & 48-Hour Patch-Test Standard:</strong> Recommendations are educational suggestions formulated from client-selected skin profiles. This AI assistant does not provide medical diagnosis. Always conduct a 48-hour patch test prior to introducing new active molecules.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 🌿 Circadian Skincare Rituals in Motion")
    st.caption("Techniques for daily application — from morning barrier awakening to nocturnal cellular replenishment:")
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.markdown(f'<div style="border-radius:18px; overflow:hidden; border:1px solid #d7c1c3; background:#f7ebed; margin-bottom:8px;"><img src="{_IMG_STEP_1}" style="display:block; width:100%; aspect-ratio:4/3; object-fit:cover;" alt="Step 1"/></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="stitch-card" style="margin-top: 8px;">
                <span class="stitch-pill stitch-pill-secondary">STEP 01 • CLEANSE</span>
                <h4 style="margin: 6px 0 4px 0; font-size: 16px;">Cleansing Emulsion Ritual</h4>
                <p style="font-size: 12.5px; color: var(--color-on-surface-variant); line-height: 1.45;">
                    Apply milky cleanser to dry skin. Gently massage in upward circular motions for 60 seconds to dissolve particulates while preserving the acid mantle.
                </p>
                <div style="font-size: 11px; color: var(--color-primary); font-weight: 600;">Standard: 60-Second Luke-Warm Rinse</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r_col2:
        st.markdown(f'<div style="border-radius:18px; overflow:hidden; border:1px solid #d7c1c3; background:#f7ebed; margin-bottom:8px;"><img src="{_IMG_STEP_2}" style="display:block; width:100%; aspect-ratio:4/3; object-fit:cover;" alt="Step 2"/></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="stitch-card" style="margin-top: 8px;">
                <span class="stitch-pill stitch-pill-secondary">STEP 02 • MICRO-INFUSION</span>
                <h4 style="margin: 6px 0 4px 0; font-size: 16px;">Cellular Serum Penetration</h4>
                <p style="font-size: 12.5px; color: var(--color-on-surface-variant); line-height: 1.45;">
                    Dispense 3-4 drops directly onto damp skin. Employ the Parisian Press & Pat technique using warm palms to drive multi-depth hyaluronic and active molecules into the epidermis.
                </p>
                <div style="font-size: 11px; color: var(--color-primary); font-weight: 600;">Technique: Damp Skin Press & Pat</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r_col3:
        st.markdown(f'<div style="border-radius:18px; overflow:hidden; border:1px solid #d7c1c3; background:#f7ebed; margin-bottom:8px;"><img src="{_IMG_STEP_3}" style="display:block; width:100%; aspect-ratio:4/3; object-fit:cover;" alt="Step 3"/></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="stitch-card" style="margin-top: 8px;">
                <span class="stitch-pill stitch-pill-secondary">STEP 03 • SEAL & NOURISH</span>
                <h4 style="margin: 6px 0 4px 0; font-size: 16px;">Lipid Barrier Lock & Glaze</h4>
                <p style="font-size: 12.5px; color: var(--color-on-surface-variant); line-height: 1.45;">
                    Warm a pea-sized amount between fingertips. Sweep upwards from jawline to temple along lymphatic drainage lines, locking in hydration with a breathable shield.
                </p>
                <div style="font-size: 11px; color: var(--color-primary); font-weight: 600;">Motion: Upward Lymphatic Sweep</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("### ☀️ Circadian AM & PM Ritual Choreography")
    active_routine = ROUTINES_BY_TYPE.get(st.session_state.selected_skin_type, ROUTINES_BY_TYPE["combination"])
    am_col, pm_col = st.columns(2, gap="large")
    with am_col:
        st.markdown(
            """
            <div class="stitch-card" style="border-left: 4px solid var(--color-primary);">
                <span class="stitch-pill stitch-pill-secondary">MORNING (AM) DEFENSE</span>
                <h4 style="margin: 6px 0 2px 0;">Morning Barrier Awakening & UV Protection</h4>
                <div style="font-size: 11px; color: #524344; margin-bottom: 12px;">4 Synchronized Steps • 5 Minutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for step in active_routine["am"]:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #d7c1c3; border-radius: 16px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--color-primary);">STEP {step['num']}</span>
                        <span class="stitch-pill" style="font-size: 9px; padding: 2px 6px;">{step['note']}</span>
                    </div>
                    <div style="font-size: 13.5px; font-weight: 700; color: #201a1c; margin-top: 3px;">{step['name']}</div>
                    <div style="font-size: 12px; color: #524344; margin-top: 2px;">{step['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with pm_col:
        st.markdown(
            """
            <div class="stitch-card" style="border-left: 4px solid var(--color-secondary);">
                <span class="stitch-pill stitch-pill-tertiary">EVENING (PM) RECOVERY</span>
                <h4 style="margin: 6px 0 2px 0;">Nighttime Cellular Renewal & Lipid Seal</h4>
                <div style="font-size: 11px; color: #524344; margin-bottom: 12px;">4 Synchronized Steps • 8 Minutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for step in active_routine["pm"]:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #d7c1c3; border-radius: 16px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--color-secondary);">STEP {step['num']}</span>
                        <span class="stitch-pill" style="font-size: 9px; padding: 2px 6px;">{step['note']}</span>
                    </div>
                    <div style="font-size: 13.5px; font-weight: 700; color: #201a1c; margin-top: 3px;">{step['name']}</div>
                    <div style="font-size: 12px; color: #524344; margin-top: 2px;">{step['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    st.markdown("### 🧴 Curated Product Formulations")
    st.caption(f"Showing formulations calibrated for **{st.session_state.selected_skin_type.capitalize()}** skin:")
    products = PRODUCTS_BY_TYPE.get(st.session_state.selected_skin_type, PRODUCTS_BY_TYPE["combination"])
    prod_col1, prod_col2 = st.columns(2, gap="medium")
    for i, prod in enumerate(products):
        with prod_col1 if i % 2 == 0 else prod_col2:
            st.markdown(
                f"""
                <div class="stitch-card" style="min-height: 240px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="stitch-pill stitch-pill-secondary" style="font-size: 9.5px;">{prod['cat']}</span>
                            <span style="font-weight: 700; color: var(--color-primary); font-size: 12.5px;">{prod['price']}</span>
                        </div>
                        <h4 style="font-size: 16px; margin: 8px 0 2px 0;">{prod['title']}</h4>
                        <span style="font-size: 10.5px; color: var(--color-secondary); font-weight: 600; text-transform: uppercase;">{prod['tag']}</span>
                        <p style="font-size: 12.5px; color: var(--color-on-surface-variant); margin: 6px 0; line-height: 1.4;">{prod['why']}</p>
                        <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:6px;">
                            {' '.join([f'<span class="stitch-pill" style="font-size:9px; padding:2px 6px;">{a}</span>' for a in prod['actives']])}
                        </div>
                        <div style="font-size: 11px; color: #524344; margin-top: 8px; font-style: italic;">
                            <strong>Ritual:</strong> {prod['ritual']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"💖 Save to Capsule — {prod['cat']}", key=f"btn_prod_{prod['id']}", use_container_width=True):
                add_to_favorites(f"Product: {prod['cat']}", prod["title"], f"{prod['why']} Actives: {', '.join(prod['actives'])}")

elif "AI Beauty Stylist" in st.session_state.active_studio_name:
    st.markdown(
        """
        <div class="stitch-hero">
            <span class="stitch-pill stitch-pill-secondary">AI BEAUTY STYLIST CONCIERGE</span>
            <h1 style="font-size: 32px; margin: 6px 0 8px 0;">Consultation with Dr. Elise Moreau Persona</h1>
            <p style="color: var(--color-on-surface-variant); font-size: 14.5px; margin: 0;">
                Ask educational questions regarding active molecule synergy, layering choreography, or routine adjustments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:12px;">
                    <div style="background:var(--color-primary); color:white; border-radius:18px 18px 4px 18px; padding:12px 18px; max-width:75%; font-size:13px; box-shadow:0 2px 8px rgba(136,74,82,0.15);">
                        {msg['text']}
                        <div style="font-size:10px; opacity:0.8; text-align:right; margin-top:4px;">{msg.get('time', '')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start; margin-bottom:12px;">
                    <div style="background:white; border:1px solid #d7c1c3; border-radius:18px 18px 18px 4px; padding:14px 18px; max-width:80%; font-size:13px; box-shadow:var(--shadow-ambient);">
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                            <span class="stitch-pill stitch-pill-secondary" style="font-size:9px; padding:1px 6px;">AI Assistant Insight</span>
                        </div>
                        <div style="color:#201a1c; line-height:1.45;">{msg['text']}</div>
                        <div style="font-size:10px; color:#857374; margin-top:4px;">{msg.get('time', '')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    user_q = st.chat_input("Ask your skincare assistant about formulations, pairings, or routines...")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "text": user_q, "time": "Now"})
        reply = (
            f"Thank you for your question regarding '{user_q}'. For your selected {st.session_state.selected_skin_type} profile, "
            "we suggest prioritizing gentle barrier support with ceramides and non-stripping cleansers, ensuring active ingredients are introduced gradually."
        )
        st.session_state.chat_messages.append({"role": "assistant", "text": reply, "time": "Now"})
        st.rerun()

elif "Saved Looks & Capsule" in st.session_state.active_studio_name:
    st.markdown(
        """
        <div class="stitch-hero">
            <span class="stitch-pill stitch-pill-secondary">CURATED ATELIER EDIT</span>
            <h1 style="font-size: 32px; margin: 6px 0 8px 0;">My Saved Looks & Product Capsule</h1>
            <p style="color: var(--color-on-surface-variant); font-size: 14.5px; margin: 0;">
                Review and export your bookmarked formulations, placement techniques, and hairstyle silhouettes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not st.session_state.favorites:
        st.info("Your capsule is empty. Click '💖 Save' on any formulation, hair silhouette, or cosmetic placement to collect it here.")
    else:
        fav_col1, fav_col2 = st.columns([2, 1], gap="large")
        with fav_col1:
            for idx, fav in enumerate(st.session_state.favorites):
                st.markdown(
                    f"""
                    <div style="background: white; border: 1px solid #d7c1c3; border-radius: 16px; padding: 14px 18px; margin-bottom: 10px; box-shadow: var(--shadow-ambient);">
                        <span class="stitch-pill stitch-pill-secondary" style="font-size: 9.5px;">{fav['category']}</span>
                        <div style="font-size: 15px; font-weight: 700; color: #201a1c; margin-top: 4px;">{fav['title']}</div>
                        <p style="font-size: 12.5px; color: #524344; margin: 4px 0 0 0; line-height: 1.4;">{fav['detail']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with fav_col2:
            if st.button("🗑️ Clear Entire Capsule", use_container_width=True):
                st.session_state.favorites = []
                st.toast("Capsule cleared.")
                st.rerun()

st.markdown("<br><hr style='border-color: #d7c1c3;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #857374; font-size: 12px; padding: 6px 0 24px 0;">
        <strong>Glow AI Haute Atelier</strong> · Skincare & Facial Architecture Studio · Designed with Google Stitch Aesthetic · All visual processing performed locally in-memory.
    </div>
    """,
    unsafe_allow_html=True,
)

