import pdfplumber

files = [
    "04A Newton Cotes.pdf",
    "04B lectura 1 Integracion_Numerica_presición_y_Error.pdf",
    "04C lectua 2 Simpson.pdf"
]

for fname in files:
    try:
        text = ""
        with pdfplumber.open(fname) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        out_name = fname + ".plumber.txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {fname} length {len(text)}")
    except Exception as e:
        print(f"Failed {fname}: {e}")
