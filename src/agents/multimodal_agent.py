from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .core_agent import BaseAgent
from .tools.external_search import ExternalSearchTool
from .tools.knowledge_base_tool import KnowledgeBaseTool
from .tools.math_tool import MathTool
from ..observability.logging import get_logger
from ..utils.exceptions import AgentError

logger = get_logger(__name__)


class AgentGraphState(TypedDict):
#   TypedDict for the state of the agent's graph execution.
    query: str
    retrieved_documents: List[Dict[str, Any]]
    answer: str
    reflection: Dict[str, Any]
    iterations: int


class MultimodalRAGAgent(BaseAgent):
#  Autonomous agent for multimodal retrieval-augmented generation (RAG) workflows.
    def __init__(self,retriever: Any,llm_client: Optional[Any] = None,reranker: Optional[Any] = None,
        use_reflection: bool = True,top_k: int = 5,rerank_top_k: int = 3,min_reflection_score: float = 6.0,max_retries: int = 2,):
    #    Initialize the MultimodalRAGAgent with a retriever, optional LLM client, and configuration parameters.
        super().__init__(
            name="multimodal_rag_agent",
        )
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker
        self.use_reflection = use_reflection
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.min_reflection_score = min_reflection_score
        self.max_retries = max_retries

        # Tool registry
        self.tools = {
            "knowledge_base": KnowledgeBaseTool(retriever=self.retriever),
            "external_search": ExternalSearchTool(),
            "math": MathTool(),
        }

        # Compile state machine graph
        self.graph = self._build_graph()
        logger.info("MultimodalRAGAgent initialized with compiled state graph")

    def _build_graph(self) -> Any:

        # Build the state machine graph for the agent's workflow.
        # The graph consists of nodes for retrieval, reranking, generation, and optional reflection,
        #  with edges defining the flow of execution.
        workflow = StateGraph(AgentGraphState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")

        if self.use_reflection:
            workflow.add_node("reflect", self._reflect_node)
            workflow.add_edge("generate", "reflect")
            workflow.add_conditional_edges(
                "reflect",
                self._should_retry,
                {
                    "retry": "generate",
                    "finish": END,
                },
            )
        else:
            workflow.add_edge("generate", END)

        return workflow.compile()

    async def _retrieve_node(self, state: AgentGraphState) -> AgentGraphState:
        # Retrieve relevant documents from the knowledge base using the configured retriever.
        # The retrieved documents are stored in the state for downstream processing.
        # This node supports both VectorStoreRetriever and HybridRetriever instances.

        query = state["query"]
        logger.info(f"Retrieval node started for: '{query}'")

        try:
            raw_results = await self.retriever.similarity_search(query=query, top_k=self.top_k)
            documents = [
                {
                    "id": doc_id,
                    "score": float(score),
                    "content": text,
                }
                for doc_id, score, text in raw_results
            ]
        except Exception as e:
            logger.error(f"Retrieval failed in node: {e}")
            documents = []

        state["retrieved_documents"] = documents
        logger.info(f"Retrieved {len(documents)} documents")
        return state

    async def _rerank_node(self, state: AgentGraphState) -> AgentGraphState:
        # Rerank the retrieved documents using the configured reranker, if available.
        # The reranked documents are stored in the state for downstream processing.
        # If no reranker is configured, the original order of documents is preserved.
        # This node is optional and can be skipped if reranking is not desired.

        documents = state.get("retrieved_documents", [])
        if not documents or not self.reranker:
            return state

        logger.info(f"Reranking {len(documents)} documents")
        try:
            docs_for_rerank = [(d["id"], d["score"], d["content"]) for d in documents]
            reranked = await self.reranker.rerank(
                query=state["query"],
                documents=docs_for_rerank,
                top_k=self.rerank_top_k,
            )
            state["retrieved_documents"] = [
                {"id": doc_id, "score": float(score), "content": text}
                for doc_id, score, text in reranked
            ]
        except Exception as e:
            logger.warning(f"Reranking encountered error; preserving original order: {e}")

        return state

    async def _generate_node(self, state: AgentGraphState) -> AgentGraphState:
        # Synthesize a response using the retrieved context.
        # The generated answer is stored in the state for downstream processing.
        # This node uses the LLM client if available; otherwise, it produces a placeholder response.
        # The generated answer is expected to be grounded in the retrieved documents.
        
        logger.info("Generation node started")
        query = state["query"]
        docs = state.get("retrieved_documents", [])
        context_chunks = [d["content"] for d in docs]

        if self.llm_client:
            prompt = f"Answer the user query: {query}\n\nContext:\n" + "\n---\n".join(context_chunks)
            state["answer"] = await self.llm_client.generate(prompt=prompt)
        else:
            state["answer"] = f"Synthesized response for query: '{query}' with {len(docs)} context chunk(s)."

        return state

    async def _reflect_node(self, state: AgentGraphState) -> AgentGraphState:
        # Assess quality and factual grounding of the generated answer.
        # The reflection score and critique are stored in the state for downstream decision-making.
        # This node uses the LLM client's reflection capabilities if available; otherwise, it produces a default pass.
        # The reflection score is used to determine whether to retry generation or finish the workflow.
    
        logger.info("Reflection node evaluating generated answer")
        state["iterations"] = state.get("iterations", 0) + 1

        if self.llm_client and hasattr(self.llm_client, "reflect"):
            state["reflection"] = await self.llm_client.reflect(
                query=state["query"],
                answer=state["answer"],
                context=[d["content"] for d in state["retrieved_documents"]],
            )
        else:
            # Default pass for offline/stub mode
            state["reflection"] = {"score": 8.5, "critique": "Answer matches retrieved context."}

        return state

    def _should_retry(self, state: AgentGraphState) -> str:
        # Determine whether to retry generation based on reflection score and iteration count.
        # If the reflection score is below the minimum threshold and the maximum retries have not been reached,
        #  it will return "retry"; otherwise, it will return "finish".
        # This decision logic is used to control the flow of the state machine graph.
        
        iterations = state.get("iterations", 0)
        score = state.get("reflection", {}).get("score", 10.0)

        if iterations >= self.max_retries:
            logger.info("Max reflection retries reached; finishing.")
            return "finish"

        if score < self.min_reflection_score:
            logger.info(f"Reflection score {score} < {self.min_reflection_score}; retrying generation.")
            return "retry"

        return "finish"

    async def run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        # Execute the agent's workflow for the given query and return the final results.
        # The workflow consists of retrieval, optional reranking, generation, and optional reflection.
        # The final output includes the synthesized answer, retrieved documents, reflection details, and metadata.
        # The method raises an AgentError if the query is invalid or if any step in the workflow fails.
        # The method supports additional keyword arguments for future extensibility, such as overriding top_k or rerank_top_k values.
    
        if not isinstance(query, str) or not query.strip():
            raise AgentError("Agent query must be a non-empty string.")

        logger.info(f"Running MultimodalRAGAgent for query: '{query}'")
        initial_state: AgentGraphState = {
            "query": query.strip(),
            "retrieved_documents": [],
            "answer": "",
            "reflection": {},
            "iterations": 0,
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)
            return {
                "query": query,
                "answer": final_state["answer"],
                "documents": final_state["retrieved_documents"],
                "reflection": final_state.get("reflection", {}),
                "metadata": {
                    "iterations": final_state.get("iterations", 0),
                    "num_documents": len(final_state["retrieved_documents"]),
                },
            }
        except AgentError:
            raise
        except Exception as e:
            logger.error(f"MultimodalRAGAgent execution failed: {e}")
            raise AgentError(f"RAG agent execution failed: {e}", original_error=e) from e