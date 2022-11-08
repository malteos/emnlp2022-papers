# emnlp2022-papers

## Download data

```bash
wget https://2022.emnlp.org/downloads/Accepted-Papers-20221027.xls
```

## Install dependecies

```bash
pip install -r requirements.txt
```

## Generate embeddings and reduce to 2D

```bash
export CUDA_VISIBLE_DEVICES=0

python embed_papers.py --input_path ./Accepted-Papers-20221027.xls \
    --output_path ./papers.json \
    --model_name_or_path /data/datasets/huggingface_transformers/pytorch/scincl --limit 10
```

## View Web page

```bash
# from local FS
open index.html

# via local web server at http://localhost
python -m http.server 80
```