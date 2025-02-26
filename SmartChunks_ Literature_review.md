# LumberChunker: Long-Form Narrative Document Segmentation 

\[Marten, A., et al. (2024). LumberChunker: Long-Form Narrative Document Segmentation. arXiv preprint arXiv:2406.17526.\] [Link](https://arxiv.org/pdf/2406.17526)

**Technique**: LumberChunker follows a three-step process. First, document is segmented paragraph-wise. Secondly, a group (G\_i) is created by appending sequential chunks until exceeding a predefined token count \\theta. Finally, G\_i is fed as context to the LLM (Gemini in their case), which determines the ID where a significant content shift starts to appear, thus defining the start of G\_{i+1} and the end of the current chunk. This process is cyclically repeated for the entire document.

**Key Components:**

* Initial Paragraph Splitting: The document is split paragraph-wise, and each paragraph is assigned a unique incremental ID.  
    
* LLM Integration: Employs transformer-based models like GPT to detect semantic boundaries.  
    
* Dynamic Segmentation: The model adapts chunk sizes based on content complexity, providing flexible segmentation.

**Benchmarking:**

- Custom benchmark GutenQA was used  
- comprises 100 books sourced from Project Gutenberg.   
- content was manually extracted to avoid errors from automatic extraction.   
- ChatGPT (gpt-3.5-turbo-0125) generated over 10,000 initial questions, later filtered to 30 high-quality factual and specific questions per book. This targeted 'what,' 'when,' and 'where' questions to evaluate retrieval capabilities, avoiding repeated content and focusing on precise information extraction. can significantly improve chunking accuracy by understanding context and semantics beyond surface-level features.

**Conclusions:**

- LLMs can improve chunking accuracy by understanding context and semantics beyond surface-level features.  
- LLMs are better suited to identify context switched information between texts compared to the distance based semantic methods like cosine similarity on embedding vectors.  
- Dynamic chunking is more effective than fixed-size chunking, especially for narrative or complex documents.

# Agentic Chunking Method I

\[Blog post by gleen.ai (No formal publication or code repository available).\]

**Technique**: Agentic chunking involves creating mini-chunks using recursive text splitting and annotating these chunks with unique markers (e.g., section headings). The LLM then groups these annotated chunks into semantically coherent sections. This approach aims to mimic human-like document structuring by providing the LLM with explicit cues.

**Key Components:**

- Recursive Text Splitting: Initial segmentation into mini-chunks based on sentence or paragraph boundaries.

- Annotation: Use of markers to guide the LLM in recognizing chunk divisions.

- LLM Grouping: The LLM is prompted to group mini-chunks into larger, semantically meaningful sections.

**Lessons Learned:**

- Annotating chunks with markers helps the LLM in identifying and maintaining document structure.

- This method shows potential but lacks empirical validation and code for experimentation.

**Conclusions**

- There is some merit to the idea to group chunks based on semantics however, the annotation step could prove a costly exercise as all the rule-based markers (e.g section headers) might not work well with *RecursiveTextSplitting* and it has to be done manually.

# Agentic Chunking Method II: Proposition-Based Chunking

\[Chen, T., et al. (2024). Dense X Retrieval: What Retrieval Granularity Should We Use? arXiv preprint arXiv:2312.06648.\] [Link](https://arxiv.org/pdf/2312.06648)

**Technique**:   
This method proposes using fine-grained units called "propositions"—atomic expressions encapsulating distinct factoids—as the basis for chunking. After breaking down text into propositions, an LLM is employed to group these propositions into semantically coherent chunks, enhancing retrieval performance.

**Key Components:**

- Proposition Extraction: Text is segmented into propositions, each representing a single factoid.

- LLM-guided grouping: The LLM analyzes and groups propositions into larger, meaningful sections.

- Retrieval Granularity: Emphasis on fine-grained retrieval units to improve retrieval accuracy.

**Lessons Learned:**

- Fine-grained units like propositions significantly outperform passage-level units in retrieval tasks.

- Combining propositional breakdown with LLM-based chunking (agentic chunking) can optimize semantic coherence and retrieval performance.

**Performance**: The method was evaluated using experiments on passage retrieval and downstream QA tasks across five datasets and six dense retrievers. Proposition-based retrieval consistently yielded higher recall and improved downstream QA performance.

# Financial Report Chunking for Effective Retrieval Augmented Generation

\[Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, Renyu Li (2024). Financial Report Chunking for Effective Retrieval Augmented Generation. arXiv:2402.05131\]

**Technique**  
The technique goes beyond traditional paragraph-level segmentation. Instead of using a fixed token or paragraph size, their method divides financial reports into meaningful structural elements (e.g., section headers, narrative texts, tables, figures, and footnotes). These elements are automatically annotated using document understanding models. By leveraging the inherent structure of financial documents, this element-based chunking method naturally produces chunks that preserve context without manual tuning of chunk sizes.

**Key Components**

- Element-Based Segmentation: Documents are dissected into constituent elements like headers, tables, figures, footnotes, and narrative texts. Each element serves as a natural segmentation boundary that reflects the document’s structure.

- Annotation with Document Understanding Models: Structural elements are identified and annotated automatically. This annotation helps guide the chunking process so that each chunk preserves key contextual information.

- Dynamic Chunk Size Determination: The approach yields optimal chunk sizes based on the natural boundaries present in the document. Eliminates the need for manual tuning of hyperparameters such as fixed token lengths.

**Lessons Learned**

- By chunking along natural document structures, the approach maintains semantic context that is often lost with uniform paragraph splitting.  
- Improved context in each chunk leads to higher retrieval accuracy, as each segment more precisely represents a coherent idea.

But,

- If documents are poorly formatted or if the structural elements are inconsistent, the method may struggle to produce effective chunks.  
- The accuracy of the underlying document understanding models is critical; misclassification of elements could result in suboptimal segmentation.  
- While highly effective for structured financial documents, the technique might require adaptation for documents with less rigid or different structural formats.

**Benchmarking**  
The authors evaluated the performance of their element-based chunking method using three primary metrics on the financebench dataset:

*Chunking Efficiency*: Measured by the total number of chunks produced. Compared against baseline methods (e.g., fixed token-length chunking) to assess the indexing and storage implications.

*Retrieval Accuracy*: Evaluated by comparing the relevance of retrieved chunks to known evidence from financial reports. Metrics such as ROUGE and BLEU scores were used to quantify how closely the retrieved text matches the ground truth.

*Q\&A Accuracy:* Assessed via both automatic evaluation (using an LLM such as GPT-4) and manual review.  
The accuracy of the final generated answers in a retrieval-augmented question-answering task served as a key indicator of the overall effectiveness of the chunking method.

```python
import numpy as np
from typing import List, Dict, Any, Optional

"""
This module demonstrates a proof-of-concept for a semantic-based chunker.

Steps:
 1. We split/segment the input text into preliminary units (e.g. paragraphs, lines, or sentences)
 2. We encode each preliminary unit with a sentence embedding model
    (using the BAAI bge-large-1.5en model via HuggingFace transformers or sentence-transformers)
 3. We merge or split these units based on their semantic similarity scores
    - We can do pairwise adjacency-based merges or cluster-based merges
 4. We decode the final chunk list back to the original textual content (with optional metadata like page numbers)

This chunker is designed so that the final step yields text chunks ready for use in your RAG pipeline.
We do not alter the rest of the pipeline—just produce new chunks that can be fed into a standard embedder.
"""

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError("Please install sentence-transformers via pip install sentence-transformers")


class SemanticChunker:
    # Consider lazy loading or embedding caching to avoid repeated model initializations for large inputs.
    _global_model = None  # class-level reference for lazy loading

    def __init__(
        self,
        model_name: str = "./BAAI_bge-large-en-v1.5",  # local path
        similarity_threshold: float = 0.6,
    ):
        """
        Initialize the chunker with a chosen embedding model.
        :param model_name: The path or name for the local BAAI/bge-large-en model.
        :param similarity_threshold: The threshold above which we merge similar chunks.
        """
        self.model_name = model_name
        self.sim_threshold = similarity_threshold

    def _lazy_load_model(self) -> SentenceTransformer:
        """
        Lazily load the SentenceTransformer model only once at the class level.
        This helps avoid repeated initializations for large volumes of data.
        """
        if SemanticChunker._global_model is None:
            # Load model from local dir
            SemanticChunker._global_model = SentenceTransformer(
                self.model_name,
                local_files_only=True  # ensure we only use local files
            )
        return SemanticChunker._global_model

    def _compute_embeddings(self, segments: List[str]) -> np.ndarray:
        """
        Encodes each segment using the specified model, returns a numpy array.
        """
        model = self._lazy_load_model()
        embeddings = model.encode(segments, convert_to_numpy=True)
        return embeddings

    def _auto_hierarchical_cut(
        self,
        embeddings: np.ndarray,
        method: str = "ward"
    ) -> List[int]:
        """
        Automatically determine the number of clusters by detecting the largest jump
        in the linkage distances, then partition the dendrogram at that distance.
        """
        import scipy.cluster.hierarchy as sch
        from scipy.cluster.hierarchy import fcluster

        Z = sch.linkage(embeddings, method=method)
        distances = Z[:, 2]
        sorted_distances = np.sort(distances)
        diffs = np.diff(sorted_distances)
        max_jump_idx = np.argmax(diffs)
        cutoff_distance = sorted_distances[max_jump_idx]
        labels = fcluster(Z, cutoff_distance, criterion='distance')
        return labels.tolist()

    def chunk(
        self,
        text_segments: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic chunking on a list of text segments.

        :param text_segments: Preliminary list of text segments.
        :param metadata: Optional parallel list of metadata dicts. Must match length.
        :return: A list of dicts with 'content' and 'metadata'.
        """
        if not text_segments:
            return []
        if metadata and len(metadata) != len(text_segments):
            raise ValueError("Metadata length must match text_segments length")
        if len(text_segments) == 1:
            return [{
                'content': text_segments[0],
                'metadata': metadata[0] if metadata else {}
            }]

        embeddings = self._compute_embeddings(text_segments)

        chunks = []
        current_chunk = text_segments[0]
        current_meta = metadata[0] if metadata else {}
        current_embedding = embeddings[0]

        for i in range(1, len(text_segments)):
            sim = self._cosine_similarity(current_embedding, embeddings[i])
            if sim >= self.sim_threshold:
                current_chunk += "\n" + text_segments[i]
                if metadata:
                    current_meta = self._merge_metadata(current_meta, metadata[i])
                current_embedding = (current_embedding + embeddings[i]) / 2.0
            else:
                chunks.append({
                    'content': current_chunk,
                    'metadata': current_meta
                })
                current_chunk = text_segments[i]
                current_meta = metadata[i] if metadata else {}
                current_embedding = embeddings[i]

        chunks.append({
            'content': current_chunk,
            'metadata': current_meta
        })

        return chunks

    def cluster(
        self,
        text_segments: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        linkage_method: str = "ward",
        num_clusters: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cluster-based approach using hierarchical clustering.
        If num_clusters is not specified, automatically detect it by scanning the largest jump.
        """
        if metadata and len(metadata) != len(text_segments):
            raise ValueError("Metadata length must match text_segments length")

        embeddings = self._compute_embeddings(text_segments)

        import scipy.cluster.hierarchy as sch
        from scipy.cluster.hierarchy import fcluster

        if num_clusters is None:
            labels = self._auto_hierarchical_cut(embeddings, method=linkage_method)
        else:
            Z = sch.linkage(embeddings, method=linkage_method)
            labels = fcluster(Z, num_clusters, criterion='maxclust')

        cluster_map = {}
        for i, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = {
                    'content': [],
                    'metadata': []
                }
            cluster_map[label]['content'].append(text_segments[i])
            if metadata:
                cluster_map[label]['metadata'].append(metadata[i])

        chunks = []
        for label, data in cluster_map.items():
            merged_text = "\n".join(data['content'])
            merged_meta = self._merge_metadata_list(data['metadata'])
            chunks.append({
                'content': merged_text,
                'metadata': merged_meta
            })
        return chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _merge_metadata(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> Dict[str, Any]:
        """Example metadata merging function. Customize as needed."""
        merged = dict(meta_a)
        for k, v in meta_b.items():
            if k in merged:
                if k == 'page_number':
                    pages = set()
                    if isinstance(merged[k], int):
                        pages.add(merged[k])
                    elif isinstance(merged[k], list):
                        pages.update(merged[k])
                    pages.add(v)
                    merged[k] = sorted(list(pages))
                else:
                    merged[k] = v
            else:
                merged[k] = v
        return merged

    def _merge_metadata_list(self, metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metas:
            return {}
        combined = dict(metas[0])
        for m in metas[1:]:
            combined = self._merge_metadata(combined, m)
        return combined


def example_usage():
    text = [
        "This is a brief introduction about cats.",
        "Cats are very popular pets.",
        "Now let's talk about nuclear physics.",
        "Dogs are also popular, they bark.",
        "Nuclear physics deals with atomic nuclei.",
        "Some details on cat breeds include Persian and Siamese.",
        "Atomic particles like protons and neutrons are relevant.",
    ]

    metadata = [
        {'page_number': 1},
        {'page_number': 1},
        {'page_number': 2},
        {'page_number': 3},
        {'page_number': 2},
        {'page_number': 4},
        {'page_number': 2},
    ]

    chunker = SemanticChunker(model_name="./BAAI_bge-large-en-v1.5", similarity_threshold=0.55)
    adjacency_chunks = chunker.chunk(text, metadata)
    print("Adjacency-based merging result:")
    for ch in adjacency_chunks:
        print(ch)
        print("-" * 40)

    # Example hierarchical approach
    cluster_chunks = chunker.cluster(text, metadata, linkage_method="ward")
    print("\nHierarchical clustering result (auto-detected clusters):")
    for ch in cluster_chunks:
        print(ch)
        print("-" * 40)


if __name__ == "__main__":
    example_usage()

class Document:
    def __init__(self, chunk: str, metadata: dict):
        self.chunk = chunk
        self.metadata = metadata


def semantic_chunkify(
    docs: List[Document],
    model_name: str = "BAAI/bge-large-en",
    similarity_threshold: float = 0.60
) -> List[Document]:
    """
    Takes in a list of Document objects (with doc.chunk and doc.metadata),
    performs adjacency-based semantic chunking, and returns new Documents
    that combine text segments (chunks) based on the specified similarity_threshold.
    """
    # 1) Extract the text segments and metadata
    text_segments = [doc.chunk for doc in docs]
    metadata_list = [doc.metadata for doc in docs]

    # 2) Perform adjacency-based semantic chunking
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold
    )
    merged_chunks = chunker.chunk(text_segments, metadata_list)

    # 3) Build new Document objects
    new_docs = []
    for merged in merged_chunks:
        new_docs.append(Document(
            chunk=merged["content"],
            metadata=merged["metadata"]
        ))
    return new_docs


def semantic_clusterify(
    docs: List[Document],
    model_name: str = "BAAI/bge-large-en",
    clustering_method: str = "agglomerative",
    num_clusters: int = 5
) -> List[Document]:
    """
    Demonstrates how to cluster text segments first, then treat each cluster as a new chunk.
    For example, 'kmeans' or 'agglomerative' can be used. 
    """
    text_segments = [doc.chunk for doc in docs]
    metadata_list = [doc.metadata for doc in docs]

    # Use the same chunker but call .cluster() instead
    chunker = SemanticChunker(model_name=model_name)
    cluster_chunks = chunker.cluster(
        text_segments,
        metadata_list,
        clustering_method=clustering_method,
        num_clusters=num_clusters
    )

    new_docs = []
    for merged in cluster_chunks:
        new_docs.append(Document(
            chunk=merged["content"],
            metadata=merged["metadata"]
        ))
    return new_docs

# Suppose you have a list of Document objects derived from PyMuPDF parsing
pdf_docs = [
    Document(chunk="Paragraph 1 text...", metadata={"page_number": 1}),
    Document(chunk="Paragraph 2 text...", metadata={"page_number": 1}),
    Document(chunk="Paragraph 3 text...", metadata={"page_number": 2}),
    # ... more ...
]

# 1) Adjacency-based merging
adj_merged_docs = semantic_chunkify(pdf_docs, similarity_threshold=0.55)
for doc in adj_merged_docs:
    print(doc.chunk)
    print(doc.metadata)
    print("-------")

# 2) Cluster-based merging
clustered_docs = semantic_clusterify(pdf_docs, num_clusters=3)
for doc in clustered_docs:
    print(doc.chunk)
    print(doc.metadata)
    print("-------")

```