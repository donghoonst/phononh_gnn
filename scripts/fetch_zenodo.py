"""List Zenodo metadata; download an explicitly selected file with checksum verification.

No extraction, no code execution. Network functionality awaits environment access.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--record",default="20196565")
    p.add_argument("--file",help="Exact filename from the listing")
    p.add_argument("--out",default="data/downloads")
    p.add_argument("--max-bytes",type=int,default=512*1024*1024)
    a=p.parse_args()
    if not a.record.isdigit():raise ValueError("Record must be numeric")
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{a.record}",timeout=60) as r:record=json.load(r)
    print(json.dumps({"title":record["metadata"].get("title"),"license":record["metadata"].get("license"),
                      "files":[{k:f.get(k) for k in ["key","size","checksum"]} for f in record.get("files",[])]},indent=2))
    if not a.file:return
    matches=[f for f in record["files"] if f["key"]==a.file]
    if len(matches)!=1:raise ValueError("Filename not unique or missing")
    f=matches[0]
    if f["size"]>a.max_bytes:raise ValueError("File exceeds --max-bytes; inspect storage before increasing")
    name=Path(a.file).name
    if name!=a.file:raise ValueError("Only plain filenames accepted")
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    target=root/name;partial=root/(name+".part")
    if target.exists() or partial.exists():raise FileExistsError(target)
    algorithm,expected=f["checksum"].split(":",1);digest=hashlib.new(algorithm)
    url=f["links"].get("self") or f["links"].get("content")
    if not url.startswith("https://zenodo.org/"):raise ValueError("Unexpected download destination")
    try:
        size=0
        with urllib.request.urlopen(url,timeout=60) as response,partial.open("xb") as output:
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>a.max_bytes:raise ValueError("Download size cap exceeded")
                digest.update(block);output.write(block)
        if size!=f["size"] or digest.hexdigest()!=expected:raise ValueError("Size/checksum mismatch")
        partial.rename(target)
        (root/(name+".provenance.json")).write_text(json.dumps(record,indent=2))
        print(f"Verified {target}: {algorithm}:{expected}")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


if __name__=="__main__":main()
