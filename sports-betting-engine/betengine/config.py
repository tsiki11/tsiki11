"""Κεντρικές ρυθμίσεις: λίγκες, πηγή δεδομένων, προεπιλογές μοντέλου και στοιχηματισμού."""
from pathlib import Path

# Κωδικοί λιγκών όπως τους χρησιμοποιεί το football-data.co.uk
LEAGUES = {
    "E0": "Premier League (Αγγλία)",
    "SP1": "La Liga (Ισπανία)",
    "I1": "Serie A (Ιταλία)",
    "D1": "Bundesliga (Γερμανία)",
    "F1": "Ligue 1 (Γαλλία)",
    "G1": "Super League (Ελλάδα)",
}
DEFAULT_LEAGUES = list(LEAGUES)

BASE_URL = "https://www.football-data.co.uk"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"

# --- Μοντέλο Dixon-Coles ---
XI = 0.0018            # time decay ανά ημέρα: βάρος = exp(-XI * ημέρες). 0.0018 -> half-life ~385 ημέρες
L2 = 1.0               # ridge penalty: "τραβάει" ομάδες με λίγα δεδομένα προς τον μέσο όρο
TRAIN_WINDOW_DAYS = 3 * 365
MAX_GOALS = 10
MIN_TEAM_MATCHES = 6   # ελάχιστοι αγώνες ομάδας στο training window για να προβλέψουμε

# --- Value betting ---
MIN_EDGE = 0.05        # ελάχιστο αναμενόμενο κέρδος: p * odds - 1 >= 5%
MAX_ODDS = 5.0         # αγνοούμε longshots (μεγάλος θόρυβος)
KELLY_FRACTION = 0.25  # fractional Kelly (1/4)
MAX_STAKE_PCT = 0.02   # ανώτατο ποντάρισμα: 2% του bankroll ανά στοίχημα
DEFAULT_MARKETS = ("OU25",)  # 1X2 ήταν αρνητικό σε όλους τους συνδυασμούς του tuning set (2024-25)
MARKET_WEIGHT = 0.6    # blending: 0 = μόνο μοντέλο, 1 = μόνο αγορά (βλ. backtest.blend_with_market)
                       # 0.6 = συντηρητική επιλογή: με 0 το backtest έδωσε ROI ~ -10%
