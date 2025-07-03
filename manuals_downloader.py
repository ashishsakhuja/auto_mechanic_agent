#!/usr/bin/env python3
import csv
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from pathlib import Path
from requests.exceptions import RequestException

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_URL      = "https://charm.li"
OUTPUT_CSV    = "charm_manifest.csv"
HEADERS       = {"User-Agent": "Mozilla/5.0"}
THROTTLE      = 0.3    # seconds between requests
YEAR_START    = 1980
YEAR_END      = 2024
# URL-encoded list of the 49 makes from charm.li
MAKES = [
    "Acura","Audi","BMW","Buick","Cadillac","Chevrolet","Chrysler","Daewoo","Daihatsu",
    "Dodge%20and%20Ram","Eagle","Fiat","Ford","Freightliner","GMC","Geo","Honda","Hummer",
    "Hyundai","Infiniti","Isuzu","Jaguar","Jeep","Kia","Land%20Rover","Lexus","Lincoln",
    "Mazda","Mercedes%20Benz","Mercury","Mini","Mitsubishi","Nissan-Datsun","Oldsmobile",
    "Peugeot","Plymouth","Pontiac","Porsche","Renault","SRT","Saab","Saturn","Scion","Smart",
    "Subaru","Suzuki","Toyota","UD","Volkswagen","Volvo","Workhorse","Yugo"
]

logging.basicConfig(level=logging.INFO, format="%(message)s")

def probe_year(make: str, year: int) -> str | None:
    """Return the /Make/Year/ URL if it gives a 200, else None."""
    url = f"{BASE_URL}/{make}/{year}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return url
    except RequestException:
        pass
    return None

def get_models_for_year(make: str, year: int, year_url: str):
    """
    Scrape the valid /Make/Year/ page for all detail links,
    then convert each to its /bundle/ equivalent.
    """
    try:
        r = requests.get(year_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except RequestException as e:
        logging.warning(f"  ✖ Couldn’t fetch {year_url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    prefix = f"/{make}/{year}/"
    entries = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if href.startswith(prefix) and not href.endswith("/bundle/"):
            model = a.get_text(strip=True)
            bundle = f"{BASE_URL}/bundle{href}"
            entries.append((model, bundle))
    return entries

def build_and_write_manifest():
    manifest = []
    for make in MAKES:
        disp = quote(make, safe="%20")  # already encoded, but ensure
        logging.info(f"→ Checking {make}")
        time.sleep(THROTTLE)

        # Probe each year
        for year in range(YEAR_START, YEAR_END + 1):
            year_url = probe_year(make, year)
            if not year_url:
                continue

            logging.info(f"  ✅ Found {make} {year}")
            entries = get_models_for_year(make, year, year_url)
            for model, bundle_url in entries:
                manifest.append({
                    "make": requests.utils.unquote(make),
                    "model": model,
                    "year": str(year),
                    "bundle_url": bundle_url
                })
            time.sleep(THROTTLE)

    # write CSV
    keys = ["make", "model", "year", "bundle_url"]
    Path(OUTPUT_CSV).unlink(missing_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest)
    logging.info(f"\n✅ Wrote {len(manifest)} entries to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_and_write_manifest()

