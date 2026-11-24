"""
api/index.py
============
Vercel serverless entry point.

Vercel routes every request to this file's `app` object and runs it as a
Python serverless function. It simply imports the Flask app from app.py.
Nothing else changes - the same code runs locally and on Vercel.
"""

import sys
import os

# make the project root importable when Vercel runs this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401
