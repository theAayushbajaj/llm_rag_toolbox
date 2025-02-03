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