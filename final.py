import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import io
import zipfile
import base64
from collections import Counter

st.set_page_config(page_title="Fundbüro KI – Pro Version", layout="wide")

@st.cache_resource(show_spinner="Lade FashionCLIP …")
def load_zero_shot_classifier():
    return pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip")

classifier = load_zero_shot_classifier()

CLOTHING_CATEGORIES = [
    "Jacke", "Hoodie", "Pullover", "T-Shirt", "Hose", "Jeans", "Rock",
    "Kleid", "Schuhe", "Sneakers", "Stiefel", "Tasche", "Rucksack",
    "Mütze", "Schal", "Handschuhe", "Sonnenbrille", "Sonstiges"
]

# ─── Session State ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = set()

st.title("🧥 Fundbüro – KI Erkennung für Kleidung")
st.caption("FashionCLIP Zero-Shot • Kategorien gruppiert • Notizen • Favoriten • Export")

# ─── Upload ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Bilder hochladen (mehrere möglich)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name in [e["filename"] for e in st.session_state.history]:
            continue

        image = Image.open(file)
        with st.spinner(f"Analysiere {file.name} …"):
            try:
                results = classifier(image, candidate_labels=CLOTHING_CATEGORIES)
                top_result = results[0]

                entry = {
                    "filename": file.name,
                    "image": image,
                    "top_label": top_result["label"],
                    "top_score": top_result["score"],
                    "all_scores": results,
                    "id": len(st.session_state.history),
                    "timestamp": pd.Timestamp.now(),
                    "note": ""  # ← neue Notiz-Feld
                }
                st.session_state.history.append(entry)
            except Exception as e:
                st.error(f"Fehler bei {file.name}: {e}")

# ─── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Übersicht", "🖼️ Galerie nach Kategorien", "⭐ Favoriten & Export"])

# ─── Tab 1: Übersicht ───────────────────────────────────
with tab1:
    if st.session_state.history:
        df = pd.DataFrame([
            {
                "Dateiname": e["filename"],
                "Kategorie": e["top_label"],
                "Sicherheit": f"{e['top_score']:.1%}",
                "Notiz": e["note"],
                "Favorit": "★" if e["id"] in st.session_state.favorites else "☆"
            }
            for e in st.session_state.history
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Bilder analysiert.")

# ─── Tab 2: Galerie gruppiert nach Kategorie ────────────
with tab2:
    if not st.session_state.history:
        st.info("Noch keine Bilder vorhanden.")
    else:
        # Häufigkeit zählen → häufigste oben
        cat_counts = Counter(e["top_label"] for e in st.session_state.history)
        sorted_cats = sorted(cat_counts.keys(), key=lambda c: cat_counts[c], reverse=True)

        for cat in sorted_cats:
            cat_items = [e for e in st.session_state.history if e["top_label"] == cat]

            with st.expander(f"📂 {cat} ({len(cat_items)} Bilder – {cat_counts[cat]}× insgesamt)"):
                # Sortier-Dropdown pro Kategorie
                sort_opt = st.selectbox(
                    f"Sortieren innerhalb '{cat}':",
                    ["Sicherheit absteigend", "Dateiname A–Z", "Neuste zuerst"],
                    key=f"sort_{cat}"
                )

                if sort_opt == "Sicherheit absteigend":
                    sorted_items = sorted(cat_items, key=lambda x: x["top_score"], reverse=True)
                elif sort_opt == "Dateiname A–Z":
                    sorted_items = sorted(cat_items, key=lambda x: x["filename"].lower())
                else:  # Neuste zuerst
                    sorted_items = sorted(cat_items, key=lambda x: x["timestamp"], reverse=True)

                cols = st.columns(3)
                for i, entry in enumerate(sorted_items):
                    col = cols[i % 3]
                    col.image(entry["image"], use_container_width=True)
                    col.markdown(f"**{entry['top_label']}** ({entry['top_score']:.1%})")
                    col.caption(entry["filename"])

                    # Notiz-Feld pro Bild
                    note_key = f"note_{entry['id']}"
                    current_note = entry["note"]
                    new_note = col.text_input("Notiz", value=current_note, key=note_key)
                    if new_note != current_note:
                        entry["note"] = new_note
                        st.rerun()

                    # Favorit-Button
                    fav_key = f"fav_{entry['id']}"
                    is_fav = entry["id"] in st.session_state.favorites
                    if col.button("★ Favorit" if is_fav else "☆ Merken", key=fav_key):
                        if is_fav:
                            st.session_state.favorites.remove(entry["id"])
                        else:
                            st.session_state.favorites.add(entry["id"])
                        st.rerun()

# ─── Tab 3: Favoriten & Export ──────────────────────────
with tab3:
    if st.session_state.favorites:
        fav_items = [e for e in st.session_state.history if e["id"] in st.session_state.favorites]
        st.write(f"{len(fav_items)} Favoriten")
        for item in fav_items:
            st.image(item["image"], width=300)
            st.write(f"**{item['top_label']}** – {item['filename']}")
            st.caption(item["note"])
    else:
        st.info("Noch keine Favoriten markiert.")

    if st.button("Export als CSV + ZIP"):
        if st.session_state.history:
            csv_data = pd.DataFrame([
                {
                    "Dateiname": e["filename"],
                    "Kategorie": e["top_label"],
                    "Sicherheit": f"{e['top_score']:.3f}",
                    "Notiz": e["note"],
                    "Top5": " | ".join([f"{r['label']} ({r['score']:.2f})" for r in e["all_scores"]])
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

            st.download_button("CSV herunterladen", csv_data, "fundstuecke.csv", "text/csv")
            st.download_button("ZIP mit Bildern", b64_zip, "fotos.zip", "application/zip")
        else:
            st.warning("Keine Daten zum Exportieren.")

if st.button("Alles löschen"):
    st.session_state.history = []
    st.session_state.favorites = set()
    st.rerun()

st.caption("Sortiert & gruppiert nach Kategorien • Persönliche Notizen möglich • 2026")
