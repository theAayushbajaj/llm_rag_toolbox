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