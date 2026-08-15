"""RAPTOR hierarchical index used by the web UI RAG paths."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans

from deepthink.chains.utility_chains import get_memory_summarizer_chain
from deepthink.runtime.bus import emit, emit_nowait


class RAPTORRetriever(BaseRetriever):
    raptor_index: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.raptor_index.retrieve(query)


class RAPTOR:
    def __init__(self, llm, embeddings_model, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.tree = {}
        self.all_nodes: dict[str, Document] = {}
        self.vector_store = None

    async def add_documents(self, documents: list[Document]):
        await emit("Step 1: Assigning IDs to initial chunks (Level 0)...")
        level_0_node_ids = []
        for i, doc in enumerate(documents):
            node_id = f"0_{i}"
            self.all_nodes[node_id] = doc
            level_0_node_ids.append(node_id)
        self.tree[str(0)] = level_0_node_ids

        current_level = 0
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            await emit(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = self._cluster_nodes(current_level_docs)

            next_level_node_ids = []
            await emit(f"Summarizing Level {next_level}...")

            summarization_tasks = []
            for i, indices in enumerate(clustered_indices):
                cluster_docs = [current_level_docs[j] for j in indices]
                summarization_tasks.append(self._summarize_cluster(cluster_docs, next_level, i))

            summaries = await asyncio.gather(*summarization_tasks)

            for summary_node in summaries:
                self.all_nodes[summary_node.metadata["id"]] = summary_node
                next_level_node_ids.append(summary_node.metadata["id"])

            self.tree[str(next_level)] = next_level_node_ids
            current_level += 1

        await emit("Step 3: Indexing all nodes with FAISS...")
        all_doc_objects = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(all_doc_objects, self.embeddings_model)
        await emit("RAPTOR Indexing complete.")

    def _cluster_nodes(self, docs: list[Document], n_clusters=None):
        import numpy as np

        embeddings = self.embeddings_model.embed_documents([d.page_content for d in docs])

        if not embeddings:
            # _cluster_nodes is synchronous; we cannot await log_stream here.
            # Use a best-effort asyncio schedule if a loop is running, else print.
            msg = (
                "WARNING: Embeddings generation returned empty. Skipping clustering for this level."
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    emit_nowait(msg)
                else:
                    emit_nowait(msg)
            except RuntimeError:
                emit_nowait(msg)
            return [list(range(len(docs)))]

        X = np.array(embeddings)

        # Heuristic for n_clusters if not provided
        if n_clusters is None:
            n_clusters = max(1, len(docs) // 5)  # Cluster size ~ 5

        if len(docs) <= 5:  # Don't cluster if too few
            return [list(range(len(docs)))]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        try:
            kmeans.fit(X)
        except ValueError as e:
            # _cluster_nodes is synchronous; we cannot await log_stream here.
            msg = f"WARNING: KMeans failed: {e}. Fallback to single cluster."
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    emit_nowait(msg)
                else:
                    emit_nowait(msg)
            except RuntimeError:
                emit_nowait(msg)
            return [list(range(len(docs)))]

        labels = kmeans.labels_

        clustered_indices = []
        for i in range(n_clusters):
            indices = np.where(labels == i)[0].tolist()
            if indices:
                clustered_indices.append(indices)
        return clustered_indices

    async def _summarize_cluster(
        self, docs: list[Document], level: int, cluster_idx: int
    ) -> Document:
        combined_text = "\n\n".join([d.page_content for d in docs])

        # Use summarization chain
        summary_chain = get_memory_summarizer_chain(self.llm)  # Reuse memory summarizer
        summary = await summary_chain.ainvoke({"history": combined_text})  # repurposing history arg

        node_id = f"{level}_{cluster_idx}"
        metadata = {
            "id": node_id,
            "level": level,
            "cluster": cluster_idx,
            "children": [d.metadata.get("id") for d in docs],
        }
        return Document(page_content=summary, metadata=metadata)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        if not self.vector_store:
            return []

        # Retrieve from full tree
        # In full RAPTOR, you might retrieve from different levels.
        # Here we just use the flattened FAISS index of all nodes.
        return self.vector_store.similarity_search(query, k=k)
