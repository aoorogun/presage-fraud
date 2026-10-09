"""Download the public datasets used by PRESAGE-Fraud into data/raw/. Licences: see research/dataset_cards/. Mirrors on Hugging Face are used
because Kaggle requires authentication; verify the upstream licence before redistributing."""
import subprocess, zipfile, pathlib
RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
FILES = {
    "paysim.csv": "https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv",
    "ibm_aml_hi_small.csv.zip": "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip",
    "elliptic_txs_classes.csv": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_edgelist.csv",
    "elliptic_txs_features.csv": "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main/elliptic_txs_features.csv",
}
for name, url in FILES.items():
    dst = RAW / name
    if not dst.exists(): subprocess.run(["curl", "-sL", "-o", str(dst), url], check=True)
    print(name, dst.stat().st_size)
z = RAW / "ibm_aml_hi_small.csv.zip"
if z.exists() and not (RAW / "HI-Small_Trans.csv").exists(): zipfile.ZipFile(z).extractall(RAW)
