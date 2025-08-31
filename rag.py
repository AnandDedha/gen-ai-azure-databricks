import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os, mlflow.deployments

def basic_llm_chat(prompt):
    """Basic LLM interaction ; I'm using databricks model you can use your own model"""
    try:
        client = mlflow.deployments.get_deploy_client("databricks")
        resp = client.predict(
            endpoint="databricks-meta-llama-3-1-8b-instruct", # choose your chat model endpoint
            inputs={"messages": [
                {"role": "system", "content": "You are a concise, accurate assistant."},{"role": "user", "content": prompt},],
                    "max_tokens": 300,
                    "temperature": 0.2,
                    },)
        return resp["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"Error calling LLM API: {str(e)}"
    
class RAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize RAG system with embedding model and vector store"""
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def add_documents(self, documents):
        """Add documents to the knowledge base"""
        self.documents.extend(documents)
        
        # Generate embeddings for all documents
        doc_embeddings = self.embedding_model.encode([doc['content'] for doc in documents])
        
        if self.embeddings is None:
            self.embeddings = doc_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embeddings])
        
        # Build FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return relevant documents with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'relevance_score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def generate_response(self, query, retrieved_docs):
        """Generate response using retrieved context"""
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['document']['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create RAG prompt
        rag_prompt = f"""Context Information:{context} Based on the above context, please answer the following question:{query}If the context doesn't contain enough information to answer the question, please say so."""
        
        # In a real implementation, this would call your LLM
        return basic_llm_chat(rag_prompt)
          
    
# Sales Data RAG
def setup_sales_rag_example():
    """Real-world example: Company sales data RAG system"""
    
    # Sample documents
    company_docs = [
        {
            "id": "sales_q1_2024", 
            "content": "Q1 2024 Sales Report: Total revenue reached $4.8M, representing 12% growth YoY. Key drivers included enterprise software sales ($2.1M) and consulting services ($1.9M). Geographic breakdown: North America 65%, Europe 25%, APAC 10%.",
            "metadata": {"type": "sales_report", "quarter": "Q1", "year": 2024}
        },
        {
            "id": "sales_q2_2024", 
            "content": "Q2 2024 Performance: Revenue hit $5.2M (+8% QoQ). Software subscriptions showed strong growth at $2.4M. New customer acquisition: 47 enterprise clients. Churn rate decreased to 3.2%.",
            "metadata": {"type": "sales_report", "quarter": "Q2", "year": 2024}
        },
        {
            "id": "product_strategy_2024", 
            "content": "2024 Product Strategy: Focus on AI-powered analytics platform. Expected market size $12B by 2025. Competitive advantage: 40% faster processing than nearest competitor. Investment needed: $2M in R&D.",
            "metadata": {"type": "strategy", "year": 2024}
        }
    ]
    
    # Initialize RAG system
    rag_system = RAGSystem()
    rag_system.add_documents(company_docs)
    
    return rag_system
# Demo the RAG system
rag_system = setup_sales_rag_example()
# Example queries
queries = [
    "What were our Q2 2024 sales figures?",
    "How much should we invest in R&D this year?",
    "What's our customer churn rate?"
]
for query in queries:
    print(f"\n Query: {query}")
    retrieved = rag_system.retrieve_context(query, top_k=2)
    
    print(" Retrieved Documents:")
    for doc in retrieved:
        print(f"  • {doc['document']['id']} (relevance: {doc['relevance_score']:.2f})")
    
    response = rag_system.generate_response(query, retrieved)
    print(f" AI Response: {response}")
