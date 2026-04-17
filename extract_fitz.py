import fitz

files = [
    "04A Newton Cotes.pdf",
    "04B lectura 1 Integracion_Numerica_presición_y_Error.pdf",
    "04C lectua 2 Simpson.pdf"
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
        
        # Test drawing the first page as png if text is empty
        if len(text.strip()) < 100:
            pix = doc[0].get_pixmap()
            pix.save(f"{fname}_page0.png")
            print(f"Saved {fname}_page0.png")
            
    except Exception as e:
        print(f"Failed {fname}: {e}")
