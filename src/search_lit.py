#!/usr/bin/env python3
"""Robust ECG literature search: Semantic Scholar (primary) + arXiv (fallback).

Respects rate limits (S2 ~1 req/s, arXiv ~1 req/3s) with retries + backoff.
Outputs compact JSON-line records to stdout and writes to a results file.
"""
import sys, os, json, time, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

NS = {'a': 'http://www.w3.org/2005/Atom'}
S2 = "https://api.semanticscholar.org/graph/v1"
ARXIV = "https://export.arxiv.org/api/query"

def _get(url, timeout=25, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'HermesAgent/1.0 (research)'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                time.sleep(5 + attempt*5)
            else:
                time.sleep(2 + attempt*2)
        except Exception as e:
            last = e
            time.sleep(3 + attempt*3)
    raise RuntimeError(f"failed after {retries}: {last}")

def s2_search(query, limit=20, fields="title,authors,year,abstract,citationCount,externalIds,openAccessPdf,venue"):
    q = urllib.parse.quote(query)
    url = f"{S2}/paper/search?query={q}&limit={limit}&fields={fields}"
    raw = _get(url)
    data = json.loads(raw)
    return data.get("data", [])

def arxiv_search(query, max_results=15, sort="relevance"):
    parts = f'all:{urllib.parse.quote(query)}'
    sort_map = {"relevance":"relevance","date":"submittedDate","updated":"lastUpdatedDate"}
    params = f"search_query={parts}&max_results={max_results}&sortBy={sort_map[sort]}&sortOrder=descending"
    raw = _get(f"{ARXIV}?{params}")
    root = ET.fromstring(raw)
    out = []
    for entry in root.findall('a:entry', NS):
        title = entry.find('a:title', NS).text.strip().replace('\n',' ')
        raw_id = entry.find('a:id', NS).text.strip().split('/abs/')[-1]
        arxiv_id = raw_id.split('v')[0]
        published = entry.find('a:published', NS).text[:10]
        authors = ', '.join(a.find('a:name', NS).text for a in entry.findall('a:author', NS))
        summary = entry.find('a:summary', NS).text.strip().replace('\n',' ')
        out.append({"arxiv_id":arxiv_id,"title":title,"year":published[:4],"authors":authors,
                    "abstract":summary,"url":f"https://arxiv.org/abs/{arxiv_id}"})
    return out

if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv)>1 else "s2"
    query = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv)>3 else 20
    if engine == "s2":
        time.sleep(0.5)
        results = s2_search(query, limit=limit)
        for r in results:
            ext = r.get("externalIds") or {}
            arxiv = ext.get("ArXiv","")
            doi = ext.get("DOI","")
            authors = ", ".join(a.get("name","") for a in (r.get("authors") or []))
            pdf = (r.get("openAccessPdf") or {}).get("url","")
            rec = {"title":r.get("title",""),"year":r.get("year"),"authors":authors,
                   "citations":r.get("citationCount",0),"arxiv":arxiv,"doi":doi,
                   "venue":r.get("venue",""),"pdf":pdf,
                   "abstract":(r.get("abstract") or "")[:500]}
            print(json.dumps(rec))
    elif engine == "arxiv":
        results = arxiv_search(query, max_results=limit)
        for r in results:
            print(json.dumps(r))
