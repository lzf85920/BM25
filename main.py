import pandas as pd
import json
import sys
import preprocessing as pre
from pyserini.search import SimpleSearcher
import argparse
import os

def save_document_json(doc_csv, out_path):
    JSON_file = []
    dic_data = {}

    index_list = doc_csv['index'].astype(str).tolist()
    doc_list = doc_csv['content'].tolist()

    for line in range(len(doc_list)):
        JSON_file.append({"id":index_list[line], "contents":doc_list[line]})

    import json
    with open(out_path, 'w') as outfile:
        for entry in JSON_file:
            json.dump(entry, outfile)
            outfile.write('\n')

def query_prepare(query_folder):
    query_list = os.listdir(query_folder)
    query_entities = {}
    for query in query_list:
      query_ent = []
      q = pre.Preprocessing(pre.extract_query(os.path.join(query_folder, str(query))))
      query_entities[query] = q.split()
    for i, j in query_entities.items():
        query_entities[i] = ' '.join([ q for q in j if len(q) > 1])
    return query_entities

def search(n):
    searcher = SimpleSearcher('./indexes/sample_collection_jsonl')
    prediction = []
    tbd_match = {}
    for top, ents in query_entities.items():
        hits = searcher.search(query_entities[top], k=n)
        pred = []
        for i in range(len(hits)):
            jsondoc = json.loads(hits[i].raw)
            pred.append(int(hits[i].docid))
          # print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f} {jsondoc["contents"][:80]}...')
        prediction.append(pred)
        tbd_match[top] = pred
    pred = pd.DataFrame(columns=['topic', 'doc'])
    for top, doc in tbd_match.items():
        doc = [str(i) for i in doc]
        top_pred = pd.DataFrame([[str(top),' '.join(doc)]], columns=['topic', 'doc'])
        pred = pred.append(top_pred)
    return pred

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--doc_csv", type=str, required=True)
    parser.add_argument("--query_folder", type=str, required=True)
    parser.add_argument("--search_output", type=str, required=True)
    args = parser.parse_args()

    os.system('mkdir ./sample_collection_jsonl')
    os.system('mkdir ./indexes')
    os.system('mkdir ./indexes/sample_collection_jsonl')
    
    doc_data = pd.read_csv(args.doc_csv)
    save_document_json(doc_data, './sample_collection_jsonl/documents.jsonl')
    query_entities = query_prepare(args.query_folder)
    
    os.system('python3 -m pyserini.index \
  --input ./sample_collection_jsonl \
  --collection JsonCollection \
  --generator DefaultLuceneDocumentGenerator \
  --index ./indexes/sample_collection_jsonl \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw')
    search_result = search(n=args.n)
    search_result.to_csv(args.search_output, index=False)














