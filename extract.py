import os
from pypdf import PdfReader

files = [
    "04A Newton Cotes.pdf",
    "04B lectura 1 Integracion_Numerica_presición_y_Error.pdf",
    "04C lectua 2 Simpson.pdf"
]

for fname in files:
    try:
        reader = PdfReader(fname)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        out_name = fname + ".txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {fname}")
    except Exception as e:
        print(f"Failed {fname}: {e}")
