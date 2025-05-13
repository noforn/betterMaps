#!/bin/bash

echo "Hol up, hol up, setting up env..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Let's map sum shit..."
python3 betterSam.py

echo "[*] LaFlame Says: generation complete, peep maps dir!"