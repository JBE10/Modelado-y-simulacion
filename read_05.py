import fitz

files = [
    "Material de clase/05A Introducción al metodo de Monte Carlo-1.pdf",
    "Material de clase/05B Repaso Monte_Carlo.pdf",
    "Material de clase/05C Lectura 1 Erraor_acotado_vs_confianza_probabilística.pdf"
]

for fname in files:
    try:
        doc = fitz.open(fname)
        text = ""
        for page in doc:
            text += page.get_text()
        
        out_name = fname + ".fitz.txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {fname} pages {len(doc)} text length {len(text)}")
    except Exception as e:
        print(f"Failed {fname}: {e}")
