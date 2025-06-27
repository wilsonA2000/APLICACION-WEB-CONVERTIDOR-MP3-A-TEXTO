# Convertidor de Audio a Texto (MP3 a TXT)

Esta aplicación web permite subir un archivo de audio (mp3, wav, m4a, ogg, flac), transcribirlo a texto usando Whisper localmente y descargar el resultado como archivo .txt.

## Requisitos
- Python 3.8+
- pip

## Instalación

1. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```
2. Ejecuta la aplicación:
   ```sh
   python app.py
   ```
3. Abre tu navegador en [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Notas
- El modelo Whisper se descarga automáticamente la primera vez que se ejecuta.
- No necesitas API keys ni pagar servicios externos.
- El archivo de texto generado se descarga automáticamente tras la transcripción.
