"""Search for the mystery folk tale with 'tawara' or 'Toda'."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("melm/contracts/folk_tales.v1.json", encoding="utf-8") as f:
    data = json.load(f)
stories = data.get("stories", [])

keywords = ["tawara", "toda", "fuji", "hidesato", "rice", "bag"]
for s in stories:
    text = s.get("text","") + " " + s.get("title","")
    for k in keywords:
        if k in text.lower():
            print(f"Found '{k}' in: {s['title'][:80]}")
            print(f"  Text[:400]: {s['text'][:400]}")
            print()
            break
else:
    print("No story found with those keywords in contract")
