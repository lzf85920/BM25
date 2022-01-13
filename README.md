# Information-Retrieval

## How to Execute

### Step 1
Download the required package
```
sh ./download_package.sh
```

### Step 2
prepare a preprocessed document csv file.

![](https://i.imgur.com/flAWEdy.png)

### Step 3
searching relevant documents

```
python3 main.py --n number of doc return --doc_csv document csv path --query_folder query folder --search_output search result csv path
```


## For example (colab) :

### Step 1

Download the required package

![](https://i.imgur.com/bplJnui.png)

### Step 2

preprocess documents, This process will take some time (12hr/100,000doc)

``` python
import preprocessing as pre

documents = ['1199555', '1208939', '1253666']
doc_csv = {}

for doc in documents:
    doc_body = pre.extract_body('/content/%s'%doc)
    predoc = pre.Preprocessing(doc_body)
    doc_csv[doc] = predoc

pd.DataFrame(list(doc_csv.items()), columns=['index', 'content'])
```

![](https://i.imgur.com/cwUfma8.png)

### Step 3

searching relevant documents

```
python3 main.py --n 50 --doc_csv /content/doc_data.csv --query_folder /content/train_query --search_output /content/output.csv
```

then you can get search result csv file

![](https://i.imgur.com/8eJm52V.png)


### Other choice
This process is faster but the accuracy will decrease (2.5hr/100,000doc). If you choose this method, You can skip converting document csv to json.

### Step 1

preprocessing documents

``` python 
import xml.etree.ElementTree as ET
import re
import string
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

docs = []
filenames = os.listdir(path_to_your_documents_collection)
for filename in filenames:
    tree = ET.parse(os.path.join('doc', filename))
    root = tree.getroot()
    body = ET.tostring(root[1], encoding='unicode', method='text')

    body = re.sub('[\[].*?[\]]', '', body)
    body = body.translate(str.maketrans('', '', string.punctuation))
    body = re.sub(' +', ' ', body).lower()

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(body)
    porter = PorterStemmer()
    body = ' '.join([porter.stem(w) for w in word_tokens if not w in stop_words])

    docs.append({'id': filename, 'contents': body})

with open('./docs.json', 'w') as f:
    json.dump(docs, f)
```

### Step 2

preprocessing query

``` python
import xml.etree.ElementTree as ET
import re
import string
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv

with open('./queries.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    queries = []
    filenames = os.listdir(path_to_your_query_folder)
    for filename in filenames:
        tree = ET.parse(os.path.join('test_query', filename))
        root = tree.getroot()
        summary = ET.tostring(root[2], encoding='unicode', method='text')

        summary = re.sub('[\[].*?[\]]', '', summary)
        summary = summary.translate(str.maketrans('', '', string.punctuation))
        summary = re.sub(' +', ' ', summary).lower()

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(summary)
        porter = PorterStemmer()
        summary = ' '.join([porter.stem(w) for w in word_tokens if not w in stop_words])

        writer.writerow([filename, summary])
```

### Step 3

Create required folders

```
mkdir /content/sample_collection_jsonl
mkdir /content/indexes
mkdir /content/indexes/sample_collection_jsonl
```


indexing documents

```
!python3 -m pyserini.index \
  --input ./sample_collection_jsonl \
  --collection JsonCollection \
  --generator DefaultLuceneDocumentGenerator \
  --index ./indexes/sample_collection_jsonl \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

### Step 4

searching 

```
!python -m pyserini.search \
    --topics ./train_query.tsv \
    --index ./indexes/sample_collection_jsonl \
    --output ./run.sample.txt \
    --bm25 -
```

``` python
output_txt = []
with open('./run.sample.txt', 'r') as f:
  for line in f.readlines():
    res = line.split()
    output_txt.append([res[0], res[2], int(res[3])])

output_df = pd.DataFrame(output_txt, columns=['topic', 'doc_index', 'num'])
output_df = output_df[output_df['num'] <= 50]
output_df
```
