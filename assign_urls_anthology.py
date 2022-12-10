import json
import pandas as pd

js_input_path = 'papers.js'
xml_input_path = '2022.emnlp.xml'
js_output_path = 'papers_with_anthology_urls.js'

with open(js_input_path) as f:
    txt = f.readlines()

# Parse paper data from JS file
papers = json.loads(txt[0][11:-2])

# Parse paper data from XNK file

import xml.dom.minidom

doc = xml.dom.minidom.parse(xml_input_path)

xml_papers = doc.getElementsByTagName("paper")
title2url = {}

for i in range(len(xml_papers)):
    paper = xml_papers[i]
    title = paper.getElementsByTagName('title')[0].toxml().replace('<title>', '').replace('</title>', '').replace('<fixed-case>', '').replace('</fixed-case>', '')

    #title = papers[i].getElementsByTagName("title")[0].firstChild.nodeValue
    #abstract = paper.getElementsByTagName("abstract")[0].firstChild.nodeValue
    url = paper.getElementsByTagName("url")[0].firstChild.nodeValue

    if title and url:
        pass
    else:
        print('error: ', paper)

    title2url[title.lower().strip()] = url

# len(xml_papers) == 834
# titles = 834


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
    
print(found) # 806

# Write back to JSON
out_txt = f'var data = {json.dumps(with_urls)}\n{txt[1]}'

with open(js_output_path, 'w') as f:
    f.write(out_txt) 
