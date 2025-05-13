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
echo "[*] Map available to view at http://localhost:$PORT/geomapbyFlame.html"
setsid python3 -m http.server "$PORT" --directory maps > /dev/null 2>&1 &
PID=$!
echo "[*] Map server PID: $PID"
echo "[*] To stop, run: kill -9 $PID"

