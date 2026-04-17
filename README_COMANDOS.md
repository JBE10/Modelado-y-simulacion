# Guía de Comandos - Modelado y Simulación

Este documento contiene las instrucciones y comandos principales para compilar y ejecutar los diferentes componentes de la materia "Modelado y Simulación" contenidos en esta carpeta.

## 📊 1. Dashboard en Python (Streamlit)

La herramienta principal para la visualización de algoritmos (Integración, Derivación, Raíces) tiene su propio script todo-en-uno que prepara el entorno (si hace falta) e inicia el servidor.

```bash
./run_dashboard.sh
```
> **¿Qué hace?** Comprueba si existe el entorno virtual (`.venv`), lo crea de lo contrario, instala/actualiza lo que esté en `requirements.txt` y automáticamente ejecuta `streamlit run dashboard.py` dentro de la subcarpeta `python/`.

---

## ⚙️ 2. Dashboard de Terminal en C / C++ (Makefile)

El segundo dashboard principal del proyecto es interactivo por consola y está desarrollado en C/C++ (carpeta `c/`). Cuentas con un archivo `Makefile` para automatizar su compilación y las rutinas subyacentes:

### Compilar el proyecto entero
```bash
make all
```
> *Compila tanto el dashboard por terminal como la API en C, creando los binarios dentro de `c/bin/`.*

### Correr el Dashboard de Terminal (TUI)
```bash
make run
```
> *Compila (sólo si no estás al día) y ejecuta el dashboard interactivo de consola (`c/bin/dashboard`).*

### Correr el Web Dashboard ligero de la API C
```bash
make run-web
```
> *Compila la lógica de la API en C (`algoritmos_api`) y luego levanta un script intermedio de Python (`c/web_dashboard.py`) para consumirla.*

### Limpiar el Entorno de Compilación
```bash
make clean
```
> *Elimina de forma segura todos los binarios y ejecutables cacheados del directorio de compilación para dejar el entorno limpio.*

---

## 🖨️ 3. Scripts de Extracción de Texto y PDF (OCR)

En la raíz encontrarás varios scripts útiles para extraer texto y apuntes de los archivos PDF del curso (como los presentes `.pdf`). Puedes invocarlos directamente con Python, asegurándote de usar tu entorno virtual preferido (ej. instalando PyMuPDF/pdfplumber):

- **Extracción rápida:** `python extract.py`
- **Extracción vía MuPDF:** `python extract_fitz.py`
- **Extracción estructural:** `python extract_plumber.py`
- **Extracción vía Imágenes (OCR):** `python ocr_mac.py`
