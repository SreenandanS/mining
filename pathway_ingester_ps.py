"""
Pathway Local FS → JSONL streamer for SARSA
- Watches a LOCAL folder in streaming mode.
- For each new/updated PDF, parses to rows using PDFAccidentParser.
- Appends rows into out/jsonl/accidents_*.jsonl for Streamlit to consume live.

Run in a separate terminal:
    python pathway_ingestor_fs.py

Config (env or edit defaults below):
    WATCH_DIR=data/dgms
    REFRESH_INTERVAL=10          # seconds
"""

import os
import io
import pathway as pw
from data_processor import PDFAccidentParser

WATCH_DIR = os.getenv("WATCH_DIR", "data/dgms")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "10"))

parser = PDFAccidentParser()

@pw.udf
def parse_pdf_to_rows(blob: bytes, path: str):
    """
    Parse each PDF into structured rows.
    Returns list[dict] compatible with your app's DataFrame schema.
    """
    import pandas as pd
    df = parser.parse_pdf(io.BytesIO(blob))
    if len(df) > 0:
        df["source_path"] = path
        # Optional: build a stable id if missing or unreliable
        # df["id"] = df.apply(lambda r: abs(hash(f"{r.get('date')}-{r.get('mine_name')}-{r.get('cause_category')}-{str(r.get('description',''))[:120]}")) % 10**9, axis=1)
    return df.to_dict("records")

# 1) Watch a local folder (recursive glob ok) in streaming mode
files = pw.io.fs.read(
    path=f"{WATCH_DIR}/**/*.pdf",
    mode="streaming",           # keep watching for changes
    format="binary",            # raw bytes
    with_metadata=True,
    refresh_interval=REFRESH_INTERVAL,
)

# 2) Parse to rows (explode per file)
rows = files.flat_map(lambda f: parse_pdf_to_rows(f.data, f._metadata.get("path", "")))

# 3) Append rows into JSONL sink for the app
pw.io.fs.write(
    rows,
    into="out/jsonl/",
    format="jsonl",
    filename_stem="accidents",
    mode="append"
)

if __name__ == "__main__":
    print(f"🚀 Pathway FS streamer watching: {WATCH_DIR} (every {REFRESH_INTERVAL}s)")
    pw.run()  # blocks & streams
