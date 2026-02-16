FROM python:3.10-slim

# Instalăm Nginx și dependințele de sistem
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalăm bibliotecile Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul codului
COPY . .

# Configurăm Nginx
COPY nginx.conf /etc/nginx/sites-available/default
# Link-ul simbolic este deja creat în majoritatea distro-urilor slim pentru 'default'

# Acordăm drepturi de execuție pentru scriptul de pornire
RUN chmod +x start.sh

# Expunem portul cerut de Hugging Face
EXPOSE 7860

# Pornim totul prin script
CMD ["./start.sh"]