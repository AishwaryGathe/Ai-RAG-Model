from phi.agent import Agent 
from phi.model.ollama import Ollama 
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.embedder.ollama import OllamaEmbedder
from phi.playground import Playground, serve_playground_app

collection_name = "thai-recipe-index"
vector_db = Qdrant(
    collection = collection_name,
    url = "http://localhost:6333/",
    embedder=OllamaEmbedder()
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db = vector_db,
)

knowledge_base.load(recreate=True, upsert=True)

agent = Agent(
    name="Thai Recipe Agent",
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
)

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("local_rag_agent:app", reload=True)
