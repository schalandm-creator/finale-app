import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import base64
import io
import zipfile

st.set_page_config(page_title="KI Bilderkennung – Pro", layout="wide")

@st.cache_resource(show_spinner="Lade Fashion-Modell …")
def load_classifier():
    return pipeline("image-classification", model="patrickjohncyh/fashion-clip")

classifier = load_classifier()

# ─── Session State ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = set()

# ─── Header ─────────────────────────────────────────────
st.title("🖼️ KI Bilderkennung – Fashion Edition")
st.caption("FashionCLIP • Multi-Upload • Vergleich • Export • Favoriten")

# ─── Upload & Verarbeitung ──────────────────────────────
uploaded_files = st.file_uploader(
    "Bilder hochladen (mehrere möglich)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name in [item["filename"] for item in st.session_state.history]:
            continue

        image = Image.open(file)
        with st.spinner(f"Analysiere {file.name} …"):
            try:
                results = classifier(image)
                top5 = results[:5]
                top_label = top5[0]["label"]
                top_score = top5[0]["score"]

                entry = {
                    "filename": file.name,
                    "image": image,
                    "top_label": top_label,
                    "top_score": top_score,
                    "top5": top5,
                    "id": len(st.session_state.history),
                    "timestamp": pd.Timestamp.now()
                }
                st.session_state.history.append(entry)
            except Exception as e:
                st.error(f"Fehler bei {file.name}: {e}")

# ─── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Ergebnisse", "🖼️ Galerie & Vergleich", "⭐ Favoriten & Export"])

with tab1:
    if st.session_state.history:
        df = pd.DataFrame([
            {
                "Dateiname": e["filename"],
                "Top Kategorie": e["top_label"],
                "Sicherheit": f"{e['top_score']:.1%}",
                "Favorit": "★" if e["id"] in st.session_state.favorites else "☆"
            }
            for e in st.session_state.history
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Bilder analysiert.")

with tab2:
    if not st.session_state.history:
        st.info("Noch keine Bilder vorhanden.")
    else:
        # Kategorien gruppieren
        categories = {}
        for entry in st.session_state.history:
            cat = entry["top_label"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry)

        sorted_cats = sorted(categories.keys())

        for cat in sorted_cats:
            with st.expander(f"📂 {cat} ({len(categories[cat])} Bilder)"):
                cols = st.columns(3)
                for i, entry in enumerate(categories[cat]):
                    col = cols[i % 3]
                    col.image(entry["image"], use_column_width=True)
                    col.markdown(f"**{entry['top_label']}** ({entry['top_score']:.1%})")
                    col.caption(entry["filename"])

                    fav_key = f"fav_{entry['id']}"
                    is_fav = entry["id"] in st.session_state.favorites
                    if col.button("★ Favorit" if is_fav else "☆ Merken", key=fav_key):
                        if is_fav:
                            st.session_state.favorites.remove(entry["id"])
                        else:
                            st.session_state.favorites.add(entry["id"])
                        st.rerun()

    if st.button("Alles löschen"):
        st.session_state.history = []
        st.session_state.favorites = set()
        st.rerun()

with tab3:
    if st.session_state.favorites:
        fav_items = [e for e in st.session_state.history if e["id"] in st.session_state.favorites]
        st.write(f"{len(fav_items)} Favoriten")
        for item in fav_items:
            st.image(item["image"], width=300)
            st.write(f"**{item['top_label']}** – {item['filename']}")
    else:
        st.info("Noch keine Favoriten markiert.")

    if st.button("Export als CSV + ZIP"):
        if st.session_state.history:
            csv = pd.DataFrame([
                {
                    "Dateiname": e["filename"],
                    "Top Label": e["top_label"],
                    "Sicherheit": f"{e['top_score']:.3f}",
                    "Top5": " | ".join([f"{r['label']} ({r['score']:.2f})" for r in e["top5"]])
                }
                for e in st.session_state.history
            ]).to_csv(index=False).encode()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for e in st.session_state.history:
                    buf = io.BytesIO()
                    e["image"].save(buf, format="PNG")
                    zf.writestr(f"{e['filename']}.png", buf.getvalue())

            zip_buffer.seek(0)
            b64_zip = base64.b64encode(zip_buffer.read()).decode()

            st.download_button("CSV herunterladen", csv, "ergebnisse.csv", "text/csv")
            st.download_button("ZIP mit Bildern", b64_zip, "bilder.zip", "application/zip", key="zip_dl")
        else:
            st.warning("Keine Daten zum Exportieren.")
