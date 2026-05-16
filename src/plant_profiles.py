"""
CIADA — Bitki Profilleri
========================
FAO [13] ve USDA-NRCS [14] kaynaklı agronomik referans değerler.

Her profil şu anahtarları içerir:
    ideals : [N_ideal (mg/kg), W_ideal (L/gün), pH_ideal]
    sigmas : [σ_N, σ_W, σ_pH]  — hassasiyet katsayıları
    bounds : [(L_N, U_N), (L_W, U_W), (L_pH, U_pH)]
"""

PLANT_PROFILES = {
    "Domates": {
        "ideals": [120.0, 4.5, 6.2],
        "sigmas": [20.0,  1.5, 0.5],
        "bounds": [(0, 250), (0, 15), (4.0, 9.0)],
        "source": "[13]",
    },
    "Misir": {
        "ideals": [180.0, 8.0, 6.8],
        "sigmas": [25.0,  2.0, 0.5],
        "bounds": [(0, 300), (0, 20), (4.0, 9.0)],
        "source": "[14]",
    },
    "Marul": {
        "ideals": [60.0, 2.5, 5.8],
        "sigmas": [15.0, 1.0, 0.4],
        "bounds": [(0, 150), (0, 10), (4.0, 9.0)],
        "source": "[13]",
    },
    "Bugday": {
        "ideals": [95.0, 4.0, 6.5],
        "sigmas": [22.0, 1.5, 0.5],
        "bounds": [(0, 250), (0, 15), (4.0, 9.0)],
        "source": "[15]",
    },
    "Biber": {
        "ideals": [75.0, 3.2, 6.0],
        "sigmas": [18.0, 1.2, 0.4],
        "bounds": [(0, 200), (0, 12), (4.0, 9.0)],
        "source": "[13]",
    },
    "Patates": {
        "ideals": [110.0, 5.5, 6.1],
        "sigmas": [28.0,  1.8, 0.5],
        "bounds": [(0, 250), (0, 15), (4.0, 9.0)],
        "source": "[16]",
    },
    "Soya": {
        "ideals": [65.0, 3.8, 6.3],
        "sigmas": [20.0, 1.3, 0.4],
        "bounds": [(0, 180), (0, 12), (4.0, 9.0)],
        "source": "[14]",
    },
    "Pamuk": {
        "ideals": [130.0, 6.2, 6.3],
        "sigmas": [30.0,  2.0, 0.5],
        "bounds": [(0, 300), (0, 18), (4.0, 9.0)],
        "source": "[13]",
    },
    "Aycicegi": {
        "ideals": [88.0, 4.1, 6.4],
        "sigmas": [22.0, 1.5, 0.5],
        "bounds": [(0, 220), (0, 14), (4.0, 9.0)],
        "source": "[13]",
    },
    "Cilek": {
        "ideals": [55.0, 2.8, 5.9],
        "sigmas": [16.0, 1.0, 0.4],
        "bounds": [(0, 150), (0, 10), (4.0, 9.0)],
        "source": "[14]",
    },
}

# Domates büyüme evresi profilleri
TOMATO_GROWTH_STAGES = {
    "Fide (0-30 gun)": {
        "ideals": [45.0,  1.5, 6.0],
        "sigmas": [12.0,  0.8, 0.4],
        "bounds": [(0, 150), (0, 8), (4.0, 9.0)],
    },
    "Vejetatif (30-60 gun)": {
        "ideals": [75.0,  2.8, 6.1],
        "sigmas": [18.0,  1.0, 0.4],
        "bounds": [(0, 200), (0, 10), (4.0, 9.0)],
    },
    "Ciceklenme (60-90 gun)": {
        "ideals": [95.0,  3.5, 6.2],
        "sigmas": [20.0,  1.2, 0.4],
        "bounds": [(0, 250), (0, 12), (4.0, 9.0)],
    },
    "Meyve (90-120 gun)": {
        "ideals": [120.0, 4.5, 6.2],
        "sigmas": [25.0,  1.5, 0.5],
        "bounds": [(0, 250), (0, 15), (4.0, 9.0)],
    },
    "Hasat (120-150 gun)": {
        "ideals": [85.0,  3.0, 6.3],
        "sigmas": [20.0,  1.0, 0.4],
        "bounds": [(0, 200), (0, 12), (4.0, 9.0)],
    },
}
