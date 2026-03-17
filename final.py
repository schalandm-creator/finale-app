import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import io
import zipfile
import base64
from datetime import datetime
from collections import Counter
from supabase import create_client, Client

# ─── Konfiguration ──────────────────────────────────────────────
st.set_page_config(page_title="Fundbüro KI – Pro", layout="wide")

# Supabase initialisieren
supabase: Client = create_client(
    st.secrets["supabase"]["url"],
    st.secrets["supabase"]["key"]
)

BUCKET_NAME = "fundstuecke"  # ← exakter Name deines Buckets!

CLOTHING_CATEGORIES = [
    "Jacke", "Hoodie", "Pullover", "T-Shirt", "Hose", "Jeans", "Rock",
    "Kleid", "Schuhe", "Sneakers", "Stiefel", "Tasche", "Rucksack",
    "Mütze", "Schal", "Handschuhe", "Sonnenbrille", "Sonstiges"
]

# ─── Modell laden ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Lade FashionCLIP …")
def load_zero_shot_classifier():
    return pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip")

classifier = load_zero_shot_classifier()

# ─── Session State (für temporäre Anzeige + Login) ─────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "history" not in st.session_state:
    st.session_state.history = []  # nur für aktuelle Session-Anzeige

# ─── Login / Logout ─────────────────────────────────────────────
with st.sidebar:
    st.header("Benutzer")
    if not st.session_state.user:
        email = st.text_input("E-Mail")
        password = st.text_input("Passwort", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.session_state.user_id = res.user.id
                    st.success("Eingeloggt!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login fehlgeschlagen: {e}")
        with col2:
            if st.button("Registrieren"):
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Registriert! Bitte E-Mail bestätigen, dann einloggen.")
                except Exception as e:
                    st.error(f"Registrierung fehlgeschlagen: {e}")
    else:
        st.write(f"Eingeloggt als: **{st.session_state.user.email}**")
        if st.button("Logout"):
            supabase.auth.sign_out()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ─── Haupt-App (nur wenn eingeloggt) ────────────────────────────
if not st.session_state.user:
    st.warning("Bitte links in der Sidebar einloggen oder registrieren.")
    st.stop()

# ─── Upload & Speichern in Supabase ─────────────────────────────
uploaded_files = st.file_uploader(
    "Bilder hochladen (mehrere möglich)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        # Doppelte vermeiden (lokal + DB prüfen)
        existing = supabase.table("fund_items") \
            .select("filename") \
            .eq("filename", file.name) \
            .eq("user_id", st.session_state.user_id) \
            .execute()
        if existing.data:
            continue

        image = Image.open(file)
        with st.spinner(f"Analysiere {file.name} …"):
            try:
                results = classifier(image, candidate_labels=CLOTHING_CATEGORIES)
                top_result = results[0]

                # Upload ins Storage
                file_ext = file.name.split('.')[-1]
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                storage_path = f"{st.session_state.user_id}/{timestamp_str}_{file.name}"

                supabase.storage.from_(BUCKET_NAME).upload(
                    path=storage_path,
                    file=file.getvalue()
                )

                # In DB speichern
                data = {
                    "user_id": st.session_state.user_id,
                    "filename": file.name,
                    "storage_path": storage_path,
                    "category": top_result["label"],
                    "confidence": top_result["score"],
                    "note": "",
                    "is_favorite": False
                }
                supabase.table("fund_items").insert(data).execute()

                # Für sofortige lokale Anzeige
                entry = {
                    "filename": file.name,
                    "image": image,
                    "top_label": top_result["label"],
                    "top_score": top_result["score"],
                    "all_scores": results,
                    "id": len(st.session_state.history),
                    "timestamp": pd.Timestamp.now(),
                    "note": "",
                    "storage_path": storage_path
                }
                st.session_state.history.append(entry)

                st.success(f"{file.name} gespeichert!")
            except Exception as e:
                st.error(f"Fehler bei {file.name}: {e}")

# ─── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Übersicht", "🖼️ Galerie nach Kategorien", "⭐ Favoriten & Export"])

# ─── Tab 1: Übersicht (aus Supabase) ────────────────────────────
with tab1:
    res = supabase.table("fund_items") \
        .select("*") \
        .eq("user_id", st.session_state.user_id) \
        .order("created_at", desc=True) \
        .execute()

    items = res.data

    if items:
        df = pd.DataFrame([
            {
                "Dateiname": item["filename"],
                "Kategorie": item["category"],
                "Sicherheit": f"{item['confidence']:.1%}",
                "Notiz": item["note"],
                "Favorit": "★" if item["is_favorite"] else "☆"
            }
            for item in items
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Fundstücke gespeichert.")

# ─── Tab 2: Galerie gruppiert ───────────────────────────────────
with tab2:
    if not items:
        st.info("Noch keine Bilder vorhanden.")
    else:
        categories = {}
        for item in items:
            cat = item["category"]
            categories.setdefault(cat, []).append(item)

        # Häufigste oben
        cat_counts = Counter(item["category"] for item in items)
        sorted_cats = sorted(categories.keys(), key=lambda c: cat_counts[c], reverse=True)

        for cat in sorted_cats:
            cat_items = categories[cat]
            with st.expander(f"📂 {cat} ({len(cat_items)} Bilder)"):
                sort_opt = st.selectbox(
                    f"Sortieren in '{cat}':",
                    ["Sicherheit ↓", "Dateiname A–Z", "Neuste zuerst"],
                    key=f"sort_{cat}"
                )

                if sort_opt == "Sicherheit ↓":
                    sorted_items = sorted(cat_items, key=lambda x: x["confidence"], reverse=True)
                elif sort_opt == "Dateiname A–Z":
                    sorted_items = sorted(cat_items, key=lambda x: x["filename"].lower())
                else:
                    sorted_items = sorted(cat_items, key=lambda x: x["created_at"], reverse=True)

                cols = st.columns(3)
                for i, item in enumerate(sorted_items):
                    col = cols[i % 3]

                    # Bild laden (public URL oder signed)
                    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(item["storage_path"])
                    col.image(public_url, use_container_width=True)

                    col.markdown(f"**{item['category']}** ({item['confidence']:.1%})")
                    col.caption(item["filename"])

                    # Notiz bearbeiten
                    note_key = f"note_{item['id']}"
                    new_note = col.text_input("Notiz", value=item["note"], key=note_key)
                    if new_note != item["note"]:
                        supabase.table("fund_items").update({"note": new_note}).eq("id", item["id"]).execute()
                        st.rerun()

                    # Favorit
                    fav_key = f"fav_{item['id']}"
                    if col.button("★ Favorit" if item["is_favorite"] else "☆ Merken", key=fav_key):
                        new_fav = not item["is_favorite"]
                        supabase.table("fund_items").update({"is_favorite": new_fav}).eq("id", item["id"]).execute()
                        st.rerun()

# ─── Tab 3: Favoriten & Export ──────────────────────────────────
with tab3:
    fav_res = supabase.table("fund_items") \
        .select("*") \
        .eq("user_id", st.session_state.user_id) \
        .eq("is_favorite", True) \
        .execute()

    fav_items = fav_res.data

    if fav_items:
        st.write(f"{len(fav_items)} Favoriten")
        for item in fav_items:
            url = supabase.storage.from_(BUCKET_NAME).get_public_url(item["storage_path"])
            st.image(url, width=300)
            st.write(f"**{item['category']}** – {item['filename']}")
            st.caption(item["note"])
    else:
        st.info("Noch keine Favoriten markiert.")

    if st.button("Export als CSV + ZIP"):
        all_res = supabase.table("fund_items") \
            .select("*") \
            .eq("user_id", st.session_state.user_id) \
            .execute()

        if all_res.data:
            csv = pd.DataFrame([
                {
                    "Dateiname": item["filename"],
                    "Kategorie": item["category"],
                    "Sicherheit": f"{item['confidence']:.3f}",
                    "Notiz": item["note"],
                    "Favorit": item["is_favorite"]
                }
                for item in all_res.data
            ]).to_csv(index=False).encode()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in all_res.data:
                    # Bild herunterladen
                    url = supabase.storage.from_(BUCKET_NAME).get_public_url(item["storage_path"])
                    response = requests.get(url)
                    zf.writestr(f"{item['filename']}.png", response.content)

            zip_buffer.seek(0)
            b64_zip = base64.b64encode(zip_buffer.read()).decode()

            st.download_button("CSV herunterladen", csv, "fundstuecke.csv", "text/csv")
            st.download_button("ZIP mit Bildern", b64_zip, "fotos.zip", "application/zip")
        else:
            st.warning("Keine Daten zum Export.")

if st.button("Alles löschen (meine Fundstücke)"):
    supabase.table("fund_items").delete().eq("user_id", st.session_state.user_id).execute()
    st.session_state.history = []
    st.session_state.favorites = set()
    st.rerun()

st.caption("Daten gespeichert in Supabase • Nur deine eigenen Fundstücke sichtbar")
