"""Build token-safe child vectors from table-aware policy passages.

Input is the cached Docling export of the four supplied PDFs, with source hashes.
No benchmark labels, expected answers or reference contexts enter this pipeline.
"""
import hashlib
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = 'HarishMaths/Hotel-Policy-Embedding'

def policy_passages(documents):
    """Keep prose paragraphs and table rows; attach every cell's column label."""
    passages = []
    removed = 0
    for doc in documents:
        blocks = re.split(r'\n\s*\n', doc['markdown'])
        section, title = '', ''
        for block in blocks:
            if block.startswith('#'):
                section = re.sub(r'^#+\s*', '', block).strip()
                title = title or section
                continue
            if section == 'Document map':
                continue
            if block.startswith('|'):
                rows = []
                for line in block.splitlines():
                    cells = [c.strip() for c in line.strip().strip('|').split('|')]
                    if all(re.fullmatch(r'[-: ]*', c) for c in cells):
                        removed += 1
                    else:
                        rows.append(cells)
                if not rows:
                    continue
                headers = rows[0]
                # Administrative policy metadata is retained on each passage,
                # rather than indexed as an answer-bearing passage.
                if headers[0] == 'Policy code':
                    continue
                for cells in rows[1:]:
                    assert len(cells) == len(headers), (doc['source'], headers, cells)
                    text = '; '.join(f'{h}: {v}' for h, v in zip(headers, cells))
                    # The qualification belongs with all training property rows.
                    if headers[0] == 'Property':
                        text += '\nTraining example only. Verify the live property profile and reservation before quoting. Amounts in US dollars, excluding tax unless noted.'
                    passages.append({'text': text, 'section': section, 'source': doc['source'], 'kind': 'table_row'})
            elif section != title and re.search(r'[A-Za-z]', block):
                passages.append({'text': re.sub(r'\s+', ' ', block).strip(), 'section': section,
                                 'source': doc['source'], 'kind': 'prose'})
    # Repeated property matrices and baseline paragraphs must not consume k.
    unique = {}
    for passage in passages:
        key = re.sub(r'\s+', ' ', passage['text']).strip()
        if key in unique:
            unique[key]['sources'].append(passage['source'])
        else:
            unique[key] = dict(passage, sources=[passage['source']])
    parents = list(unique.values())
    for i, parent in enumerate(parents):
        parent['parent_id'] = str(i)
        parent['sources'] = sorted(set(parent['sources']))
    return parents, removed, len(passages)-len(parents)

def chunk_documents(documents, tokenizer, max_seq_length=128):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=110, chunk_overlap=20,
        separators=['\n\n', '\n', '. ', '; ', ' ', ''])
    parents, removed, duplicates = policy_passages(documents)
    children = []
    for parent in parents:
        complete = parent['section'] + '\n' + parent['text']
        for part in splitter.split_text(complete):
            assert re.search(r'[A-Za-z0-9]', part), 'Empty or separator-only child'
            count = len(tokenizer.encode(part))
            assert count <= max_seq_length, count
            metadata = {k: parent[k] for k in ['source','sources','section','kind','parent_id']}
            metadata.update(parent_text=complete, tokens=count, policy_effective='2026-01-01')
            children.append(Document(page_content=part, metadata=metadata))
    report = {'vector_count': len(children), 'parent_count': len(parents),
              'separator_rows_removed': removed, 'duplicate_parents_removed': duplicates,
              'separator_only_chunks': 0, 'max_tokens': max(d.metadata['tokens'] for d in children),
              'min_tokens': min(d.metadata['tokens'] for d in children),
              'parents': parents, 'chunks': [{'text':d.page_content, 'metadata':d.metadata,
                                             'tokens':d.metadata['tokens']} for d in children]}
    return children, report

def build(documents, embeddings, output):
    children, report = chunk_documents(documents, embeddings._client.tokenizer,
                                       embeddings._client.max_seq_length)
    vectorstore = FAISS.from_documents(children, embeddings)
    vectorstore.save_local(str(output))
    Path(output).parent.joinpath('index_build_report.json').write_text(json.dumps(report,indent=2))
    return vectorstore, report

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='policy_markdown.json')
    parser.add_argument('--output',default='faiss_index')
    args=parser.parse_args()
    embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device':'cpu'},
                                    encode_kwargs={'normalize_embeddings':True})
    _, report=build(json.loads(Path(args.input).read_text()), embeddings, Path(args.output))
    print({k:v for k,v in report.items() if k not in ['parents','chunks']})
