# backend/testing/__init__.py
"""
WISTA AI Agent - Testing Framework
====================================
Framework evaluasi otomatis untuk AI Agent Travel Itinerary WISTA.

Struktur:
    scenarios.py        - Definisi 5 skenario uji dari test_prompts.txt
    evaluator.py        - Logic evaluasi (tool check, argumen check, quality check)
    runner.py           - Test runner utama
    report_generator.py - Generate laporan hasil evaluasi (tabel + JSON)
    reports/            - Output folder hasil evaluasi

Cara jalankan:
    python -m backend.testing.runner
    python backend/testing/runner.py
"""
