"""Prepare factual catalog/PDF embeddings once, using the existing GMS key."""
from pathlib import Path
import argparse,hashlib,json,os,shutil
from dotenv import load_dotenv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-file",type=Path,required=True)
    parser.add_argument("--copy-seed-to",type=Path)
    args=parser.parse_args()
    os.environ["SCENE_PUBLIC"]="1"
    load_dotenv(args.env_file,override=False)
    from scene_source.config import load_settings
    from scene_source.data import load_movies,movie_text
    from scene_source.pdf_sources import ensure_catalog,records
    from scene_source.store import split_document
    from scene_source.remote_embeddings import GMSRemoteEmbeddings,PUBLIC_MODEL,SMOKE_QUERIES,embedding_key
    root=Path(__file__).resolve().parent
    if not (root/"public_data/movies.json").exists():raise RuntimeError("Run from the exported public checkout")
    settings=load_settings(root);movies=load_movies(root)
    movie=next(m for m in movies if str(m["movie_id"])=="157336")
    ensure_catalog(settings,movie)
    texts=[movie_text(m) for m in movies]
    pdf_inputs=0
    for source in records(root):
        if not source.get("catalog_id"):continue
        for line in (settings.data_dir/"pdf_sources"/(source["id"]+".pages.jsonl")).read_text(encoding="utf-8").splitlines():
            doc=json.loads(line)
            for chunk in split_document(doc["text"],settings.chunk_size,settings.chunk_overlap):
                texts.append(f"문서: {doc['title']}\n{chunk}");pdf_inputs+=1
    texts.extend(SMOKE_QUERIES)
    texts=list(dict.fromkeys(texts))
    embedding=GMSRemoteEmbeddings(settings)
    vectors=[]
    for start in range(0,len(texts),128):
        vectors.extend(embedding.documents(texts[start:start+128]))
        print("Prepared embedding inputs",min(start+128,len(texts)),"/",len(texts),flush=True)
    seed=root/"public_data/embeddings.jsonl"
    manifest=root/"public_data/embedding_manifest.json"
    previous=json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    with seed.open("w",encoding="utf-8",newline="\n") as stream:
        for text,vector in zip(texts,vectors):
            stream.write(json.dumps({"key":embedding_key(text),"vector":vector},separators=(",",":"))+"\n")
    report={"model":PUBLIC_MODEL,"dimensions":384,"inputs":len(texts),"catalog_inputs":len(movies),
        "official_pdf_inputs":pdf_inputs,"smoke_query_inputs":len(SMOKE_QUERIES),
        "sha256":hashlib.sha256(seed.read_bytes()).hexdigest(),
        "actual_gms_calls_this_run":embedding.calls,"actual_usage_this_run":embedding.usage,
        "prepared_api_calls_total":previous.get("prepared_api_calls_total",previous.get("actual_gms_calls_this_run",0))+embedding.calls,
        "prepared_usage":previous.get("prepared_usage",previous.get("actual_usage_this_run",[]))+embedding.usage,
        "notice":"Factual public catalog, official NASA text chunks and two public smoke-query vectors. No API key or course archive."}
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.copy_seed_to:
        args.copy_seed_to.mkdir(parents=True,exist_ok=True)
        for path in (seed,manifest):shutil.copy2(path,args.copy_seed_to/path.name)
    print(json.dumps({k:v for k,v in report.items() if k not in {"actual_usage_this_run","prepared_usage"}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
