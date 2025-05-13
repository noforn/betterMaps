#!/bin/bash

echo "Hol up, hol up, setting up env..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Let's map sum shit..."
python3 betterSam.py
echo ""
echo "[*] LaFlame Says: generation complete, peep maps dir!"
echo ""
PORT=8000
cd maps && python3 -m http.server $PORT 
echo "[*] Map available to view at http://localhost:$PORT/geomapbyFlame.html"
echo ""
