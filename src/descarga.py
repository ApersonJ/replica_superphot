import time
from pathlib import Path
import pandas as pd
from alerce.core import Alerce

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
COLS = ["mjd", "fid", "magpsf", "sigmapsf", "ra", "dec", "isdiffpos"]
REINTENTOS = 3

client = Alerce()
nombres = pd.read_csv("data/muestra.csv")["NAME"]
fallos = []

for i, oid in enumerate(nombres):
    destino = RAW / f"{oid}.parquet"
    if destino.exists():                 
        continue
    for intento in range(REINTENTOS):
        try:
            det = client.query_detections(oid, survey="ztf", format="pandas")
            det[COLS].to_parquet(destino, index=False)
            break
        except Exception as e:
            if intento == REINTENTOS - 1:
                fallos.append((oid, str(e)))
            else:
                time.sleep(2 * (intento + 1))
    if i % 100 == 0:
        print(i, "de", len(nombres), flush=True)

pd.DataFrame(fallos, columns=["NAME", "error"]).to_csv("data/fallos_descarga.csv", index=False)
print("listo. fallos:", len(fallos))