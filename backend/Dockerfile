
FROM python:3.13.15-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/huggingface

# Download the model at build time
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# Cache the fixed cross-encoder used to rerank hybrid retrieval candidates.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu', max_length=512)"

# Force offline mode at runtime so it uses the cached model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
