import streamlit as st
import csv
import io
from motor import planla

st.set_page_config(page_title="Sefer Planlama", layout="centered")

st.title("🚌 Sefer Planlama Motoru")
st.write("CSV yükle → optimize et → indir")

uploaded = st.file_uploader("CSV dosyası yükle", type="csv")

if uploaded:
    text = io.TextIOWrapper(uploaded, encoding="ISO-8859-9")
    rows = list(csv.DictReader(text, delimiter=";"))

    if st.button("Optimize Et"):
        tekci, normalci = planla(rows)

        kart_map = {}
        k = 1
        for c in normalci:
            for r in c:
                kart_map[id(r)] = f"N{k}"
            k += 1

        k = 1
        for c in tekci:
            for r in c:
                kart_map[id(r)] = f"T{k}"
            k += 1

        for r in rows:
            r["KART"] = kart_map.get(id(r), "")

        output = []
        output.append(";".join(rows[0].keys()))
        for r in rows:
            output.append(";".join(r.values()))

        st.download_button(
            "⬇️ Sonucu indir",
            "\n".join(output),
            file_name="cikti.csv",
            mime="text/csv"
        )
