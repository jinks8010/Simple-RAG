import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
        self.client = QdrantClient(":memory:")
        self.collection_name = "rag_collection"

        # LLM setup
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.llm_pipeline = pipeline("text2text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer)
    

    def load_from_pdf(self, pdf_file):
        self.documents = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    self.documents.extend([s.strip() for s in sentences if s.strip()])

        self.create_index()
    

    def create_index(self):
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True).tolist()

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=len(self.embeddings[0]), distance=Distance.COSINE)
        )

        points = [
            PointStruct(id=i, vector=self.embeddings[i], payload={"text": self.documents[i]})
            for i in range(len(self.documents))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
    

    def retrieve(self, query, top_k):
        query_embedding = self.model.encode([query])[0].tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return "\n".join([hit.payload['text'] for hit in results])
    

    def generate_answer(self, query, top_k):
        context = self.retrieve(query, top_k=top_k)
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"
        result = self.llm_pipeline(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
        return result
