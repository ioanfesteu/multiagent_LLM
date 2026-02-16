#!/bin/bash

# 1. Pornim Solara pe un port intern (8765)
# app.py va porni automat și serverul Flask pe portul 5000 datorită thread-ului tău
solara run app.py --host 127.0.0.1 --port 8765 &

# 2. Pornim Nginx în foreground pentru a menține containerul activ
nginx -g "daemon off;"