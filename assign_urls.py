import json
import pandas as pd

js_input_path = 'papers.js'
csv_input_path = 'EMNLP22 preprints - emnlp2022-accept-skim.csv'
js_output_path = 'papers_with_urls.js'

with open(js_input_path) as f:
    txt = f.readlines()

# Parse paper data from JS file
papers = json.loads(txt[0][11:-2])

# Load preprint URLS (len=731)
df = pd.read_csv(csv_input_path)

title2url = {}
for idx, row in df.iterrows():
    title2url[row['Title'].lower().strip()] = row['url']

found = 0 
with_urls = []
for p in papers:
    t = p['title'].lower().strip()
    if t in title2url:
        u = title2url[t]
        found += 1
    else:
        u = ''

    p['url'] = u

    with_urls.append(p)
    
print(found) # 725

# Write back to JSON
out_txt = f'var data = {json.dumps(with_urls)}\n{txt[1]}'

with open(js_output_path 'w') as f:
    f.write(out_txt) 
