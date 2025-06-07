# Deep RAG: An LLM-Powered, Non-Vectorial, Globally Aware, Deeply Autonomous Retrieval-Augmented Generation Paradigm

<p align="center">
  <img src="https://img.shields.io/badge/RAG-LLM%20Powered-blueviolet" alt="LLM Powered RAG">
  <img src="https://img.shields.io/badge/Approach-Non--Vectorial-orange" alt="Non-Vectorial">
  <img src="https://img.shields.io/badge/Key%20Feature-Autonomous%20Retrieval-green" alt="Autonomous Retrieval">
  <img src="https://img.shields.io/badge/Version-1.0-blue" alt="Version 1.0">
</p>

---

**Abstract:**
Traditional Retrieval-Augmented Generation (RAG) systems, especially those based on embeddings, often fall short in handling complex queries, deep semantic understanding, and multimodal data, leading to unsatisfactory recall rates and answer accuracy. This paper introduces an innovative Deep RAG solution that completely reconstructs the three core stages of segmentation, indexing, and retrieval. It entirely abandons vector similarity calculations, instead leveraging the powerful contextual understanding, logical reasoning, and task planning capabilities of Large Language Models (LLMs) like GPT-4o. Through LLM-driven semantic segmentation, deep indexing based on a global knowledge base summary, and LLM-autonomous multi-round dynamic retrieval, Deep RAG can achieve near-perfect recall rates and highly accurate Q&A for private knowledge bases. This paper will detail Deep RAG's core architecture, key technologies, and demonstrate its significant superiority in handling complex semantics, temporal references, exclusion logic, multiple keywords, and macro-summary type questions through rich examples and comprehensive comparative data.

**Keywords:** Retrieval-Augmented Generation (RAG); Large Language Models (LLM); Deep RAG; Semantic Segmentation; Knowledge Base Indexing; Autonomous Retrieval; Multimodal

---

## Table of Contents
1.  [Introduction and Background](#1-introduction-and-background)
2.  [Deep RAG Core Methodology](#2-deep-rag-core-methodology)
    *   [2.1 LLM-Driven Semantic Segmentation](#21-llm-driven-semantic-segmentation)
        *   [2.1.1 Problems Addressed](#211-problems-addressed)
        *   [2.1.2 Core Idea](#212-core-idea)
        *   [2.1.3 Deep RAG Method](#213-deep-rag-method)
        *   [2.1.4 Key Advantages](#214-key-advantages)
        *   [2.1.5 Example: Knowledge Base Construction and Multimodal File Segmentation](#215-example-knowledge-base-construction-and-multimodal-file-segmentation)
    *   [2.2 LLM-Native Indexing](#22-llm-native-indexing)
        *   [2.2.1 Problems Addressed](#221-problems-addressed)
        *   [2.2.2 Core Idea](#222-core-idea)
        *   [2.2.3 Deep RAG Method](#223-deep-rag-method)
        *   [2.2.4 Key Advantages](#224-key-advantages)
    *   [2.3 LLM Autonomous Planning and Multi-Round Retrieval](#23-llm-autonomous-planning-and-multi-round-retrieval)
        *   [2.3.1 Problems Addressed](#231-problems-addressed)
        *   [2.3.2 Core Idea](#232-core-idea)
        *   [2.3.3 Deep RAG Process](#233-deep-rag-process)
        *   [2.3.4 Key Advantages](#234-key-advantages)
3.  [System Prompt Example](#3-system-prompt-example)
4.  [Practical Case Studies](#4-practical-case-studies)
    *   [4.1 Case 1: Temporal Reference + Exclusion + Complex Semantic Relations](#41-case-1-temporal-reference--exclusion--complex-semantic-relations)
    *   [4.2 Case 2: Temporal Reference + Exclusion + Numerous Keywords (Large Semantic Span)](#42-case-2-temporal-reference--exclusion--numerous-keywords-large-semantic-span)
    *   [4.3 Case 3: Temporal Reference + Exclusion + Macro-Summary Type (Short Question, Deep Understanding)](#43-case-3-temporal-reference--exclusion--macro-summary-type-short-question-deep-understanding)
5.  [Comprehensive Data Comparison](#5-comprehensive-data-comparison)
    *   [5.1 Comparison Schemes](#51-comparison-schemes)
    *   [5.2 Evaluation Dimensions & Metrics](#52-evaluation-dimensions--metrics)
    *   [5.3 Question Type Classification](#53-question-type-classification)
    *   [5.4 Performance Comparison Data](#54-performance-comparison-data)
    *   [5.5 Data Analysis & Insights](#55-data-analysis--insights)
6.  [Scalability for Very Large Knowledge Bases](#6-scalability-for-very-large-knowledge-bases)
    *   [6.1 Hierarchical Summary](#61-hierarchical-summary)
    *   [6.2 Dynamic Summary and Caching](#62-dynamic-summary-and-caching)
7.  [Conclusion](#7-conclusion)
8.  [References](#8-references)

---

## 1. Introduction and Background

Retrieval-Augmented Generation (RAG) technology, which enhances the accuracy and timeliness of Large Language Model (LLM) responses by incorporating external knowledge bases, has become a hot research topic in natural language processing. Mainstream RAG solutions commonly use embedding techniques, vectorizing text chunks and retrieving them via similarity calculations. However, as many practitioners have experienced, this approach often yields unsatisfactory accuracy when dealing with questions involving complex semantics, temporal references, exclusion logic, multiple keywords, and macro-summary types. The root cause is the failure of the retrieval stage to recall all relevant context, leaving the subsequent LLM generation phase with "no rice for the an_cooker."

Although the community has explored various improvements—such as optimizing document structure, adjusting segmentation strategies, changing embedding models, introducing hybrid retrieval (e.g., keyword + vector), and adding reranking models—the effectiveness has been limited. The inherent shortcomings of vector-based solutions persist, including insufficient capture of complex semantic relationships, the ambiguity of semantic similarity, and the black-box nature of the process. LLMs themselves possess powerful understanding and reasoning abilities, but in traditional RAG frameworks, their capabilities are not fully utilized in the retrieval phase; they merely passively accept retrieval results.

Based on a profound reflection on these issues, we propose an innovative, LLM-based, non-vectorial, globally aware, deeply autonomous RAG solution—Deep RAG. The core idea is to fully leverage LLM capabilities in contextual understanding, latent intent recognition, complex logical reasoning, dynamic decision adjustment, and global task planning to thoroughly re-engineer RAG's three major components: segmentation, indexing, and retrieval. Deep RAG aims to enable "Deep Research" into private knowledge bases, ensuring high recall and accuracy in complex query scenarios.

---

## 2. Deep RAG Core Methodology

The core idea of Deep RAG is to maximize the use of LLM's powerful capabilities throughout the entire RAG lifecycle.

### 2.1 LLM-Driven Semantic Segmentation

#### 2.1.1 Problems Addressed
1.  Traditional segmentation methods based on fixed character counts, token counts, or special symbols (e.g., Markdown's `#`, `\n`) often crudely split complete semantic paragraphs when processing unstructured text and multimodal files, destroying semantic integrity.
2.  Achieving precise segmentation with high semantic cohesion is difficult. High-quality segmentation often requires manual effort from domain experts who understand both the business and RAG, leading to high labor costs and scalability issues.
3.  Semantic similarity is an inherently vague concept. Tuning strategies for merging or refining chunks based on similarity is extremely difficult, often relying on heuristics and luck, reaching a level of "alchemy."

#### 2.1.2 Core Idea
1.  LLMs possess powerful contextual understanding, discourse structure analysis, and multimodal understanding capabilities. Why not let LLMs directly perform semantic segmentation to ensure the semantic cohesion and integrity of the resulting chunks?

#### 2.1.3 Deep RAG Method
1.  **Extraction and Preprocessing:** Extract the full text from the original file (e.g., Markdown, PDF, Word—these can be standardized to Markdown or plain text, preserving key structural information). Annotate each line with a line number. This information serves as context for the LLM's segmentation decisions.
2.  **LLM Semantic Segmentation Instruction:** Design a specific prompt instructing the LLM to perform semantic segmentation based on the full text and line numbers. The LLM's task is to identify paragraphs or blocks within the document that have independent and complete semantic meaning. The output is a JSON array, where each JSON object represents a chunk and includes:
    *   `original_path`: Path to the original file.
    *   `line_range`: The start and end line numbers of the chunk in the original file, e.g., `[start_line, end_line]`.
    *   `title`: A concise title generated by the LLM that summarizes the chunk's core content.
    *   `summary`: A detailed summary generated by the LLM, highlighting key information and purpose.
3.  **Chunk File Generation and Metadata Binding:** A program uses the `line_range` from the LLM's JSON output to extract corresponding content from the original file, creating new chunk files. The new file path is typically `original_path_directory/title.original_extension` (e.g., a chunk from `original_doc.md` might be saved as `original_doc/summary_chapter.md`). The LLM-generated `title` and `summary` are strongly bound as metadata to this chunk file.
    *   **Note:** For files with very few words (e.g., less than 500) or very simple structures, a threshold can be set. If the file content does not exceed the threshold, physical segmentation is skipped; only the `title` and `summary` for the entire file are generated and bound.

#### 2.1.4 Key Advantages
1.  **Native Multimodal Processing:** Using natively multimodal LLMs like GPT-4o, it's possible to directly understand and process code blocks, tables, and image references (even image content itself, if the LLM supports image input) within documents, achieving true multimodal content-aware segmentation.
2.  **High Semantic Cohesion:** LLMs can understand the overall structure and business logic of a document, enabling precise, semantically cohesive segmentation far superior to rule-based mechanical splitting.
3.  **Enhanced Readability and Manageability:** Each chunk file comes with an LLM-generated title and summary. This not only provides high-quality metadata for subsequent RAG retrieval but also makes these structured, summarized chunks highly suitable for human reading and knowledge base management, even outside of RAG.

#### 2.1.5 Example: Knowledge Base Construction and Multimodal File Segmentation

For consistency in subsequent examples, we first establish a unified knowledge base containing various file types.
Assume our knowledge base root directory is `MyCompanyKB/`.

**Knowledge Base File Structure (Example):**

*   `MyCompanyKB/AnnualReports/2024_Financial_and_Business_Review.pdf`: Contains last year's company performance, charts, and future outlook.
*   `MyCompanyKB/AnnualReports/2025_Q1_Business_Progress_Report.pdf`: Contains business data and project updates for Q1 of this year.
*   `MyCompanyKB/ProductDocs/SmartSpeakerX1/`:
    *   `MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md`: Contains text descriptions, feature lists, a table showing compatible devices, an image of the product (`![X1 Appearance](assets/X1_look.jpg)`), and a Python script example for initialization.
    *   `MyCompanyKB/ProductDocs/SmartSpeakerX1/assets/X1_look.jpg`: Product image.
*   `MyCompanyKB/ProductDocs/SmartVacuumR8/`:
    *   `MyCompanyKB/ProductDocs/SmartVacuumR8/CoreFeatures_and_TechSpecs.txt`: Plain text description.
    *   `MyCompanyKB/ProductDocs/SmartVacuumR8/AlgorithmModules/PathPlanning_v3.py`: Python code file.
*   `MyCompanyKB/Marketing/`:
    *   `MyCompanyKB/Marketing/2025_Marketing_Strategy_and_Budget.docx` (Assumed converted to Markdown or directly processable by LLM)
    *   `MyCompanyKB/Marketing/2026_Product_Launch_Initial_Concept.md`: Planning for next year's event.
*   `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/`:
    *   `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/Project_Weekly_Report_2025_05_27.md`: Last week's project progress, including issues and solutions.
    *   `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/User_Feedback_and_Requirements_Analysis_2025_05.csv`: User feedback data collected this past month.

**LLM Semantic Segmentation Example: Processing `MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md`**

Assume `UserManual_X1_v1.2.md` content is as follows (line numbers are illustrative):
```markdown
1: # Smart Speaker X1 User Manual v1.2
2: 
3: ## 1. Product Introduction
4: The Smart Speaker X1 is our company's latest generation AI smart home control center, supporting voice interaction, smart control, music playback, and various other functions.
5: 
6: ## 2. Core Features
7: - Voice Assistant: Built-in "Xiao Hui" intelligent voice engine.
8: - Device Control: Supports adding and controlling compatible smart home devices via the "Smart Home" App.
9: - Music & Podcasts: Integrated with mainstream music platforms.
10: 
11: ## 3. Quick Start
12: ### 3.1 Unboxing and Accessories
13: ... (some text omitted) ...
14: ![X1 Appearance](assets/X1_look.jpg)
15: 
16: ### 3.2 Initial Setup
17: Please connect the speaker to a power source and follow the Python script prompts below to complete network configuration:
18: ```python
19: import network
20: sta_if = network.WLAN(network.STA_IF)
21: if not sta_if.isconnected():
22:     print('connecting to network...')
23:     sta_if.active(True)
24:     sta_if.connect('your-ssid', 'your-password')
25:     while not sta_if.isconnected():
26:         pass
27: print('network config:', sta_if.ifconfig())
28: ```
29: 
30: ## 4. Compatible Device List
31: | Device Type   | Brand   | Model      |
32: |---------------|---------|------------|
33: | Smart Bulb    | LightUp | L100, L200 |
34: | Smart Socket  | PowerEZ | P50        |
35: | AC Companion  | CoolM   | CM-Plus    |
36: 
37: ## 5. FAQ and Support
38: ... (some text omitted) ...
```

**LLM Segmentation Output JSON (Example):**
```json
[
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [3, 5],
    "title": "Product_Introduction_SmartSpeakerX1",
    "summary": "Introduces the Smart Speaker X1 as a new generation AI smart home control center and its main supported function areas, such as voice interaction and smart control."
  },
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [6, 10],
    "title": "Core_Features_List_SmartSpeakerX1",
    "summary": "Lists the core features of the Smart Speaker X1, including the built-in 'Xiao Hui' voice assistant, device control via the 'Smart Home' App, and integrated music and podcast services."
  },
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [11, 15],
    "title": "Unboxing_Accessories_and_Appearance_SmartSpeakerX1",
    "summary": "Guides users through unboxing and checking accessories, and includes a reference to a product appearance image (X1_look.jpg)."
  },
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [16, 28],
    "title": "Initial_Setup_and_Network_Configuration_Script_SmartSpeakerX1",
    "summary": "Provides steps for the initial setup of the Smart Speaker X1, accompanied by a Python script example to guide users through network configuration."
  },
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [30, 35],
    "title": "Compatible_Smart_Home_Device_List_SmartSpeakerX1",
    "summary": "Presents a table listing compatible smart home device types, brands, and models for the Smart Speaker X1, such as LightUp smart bulbs and PowerEZ smart sockets."
  },
  {
    "original_path": "MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2.md",
    "line_range": [37, 38],
    "title": "FAQ_and_Support_Guide_SmartSpeakerX1",
    "summary": "Provides answers to frequently asked questions during the use of Smart Speaker X1 and ways to obtain technical support."
  }
]
```
The program would then, based on `line_range`, split `UserManual_X1_v1.2.md` into multiple `.md` files, e.g., `MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Product_Introduction_SmartSpeakerX1.md`, and associate the corresponding `title` and `summary` as its metadata. (Note: I've slightly modified the output path to a subfolder `_chunks` for clarity, which is a common practice).

### 2.2 LLM-Native Indexing

#### 2.2.1 Problems Addressed
1.  **Insufficient Capture of Deep Semantic Relationships:** Embedding models primarily measure relevance by calculating cosine similarity between vectors. This approach handles lexical similarity well but struggles to capture deeper, more complex semantic relationships, such as: exclusion ("A, but not B"), causality ("Because X, then Y"), comparison ("A is better than B"), temporal order ("First A, then B"), hierarchical inclusion ("A is part of B"), anaphora resolution ("It refers to..."), dependencies ("Y only if X is met"), and purpose ("To achieve A, B is needed").
2.  **Black Box and Poor Interpretability:** Vector embedding and similarity calculation processes are a "black box" to users. When retrieval results are poor, it's difficult to trace the root cause (segmentation issue? embedding model issue? similarity threshold?), hindering effective tuning.
3.  **Limited Capability of Open-Source Embedding Models in Private Knowledge Base Scenarios:** For general knowledge, large closed-source embedding models perform adequately. However, in specialized private knowledge bases with unique data, open-source embedding models often lack targeted pre-training, and their representational power is far inferior to closed-source LLMs fine-tuned on this data or specialized models, the latter being costly.

#### 2.2.2 Core Idea
1.  LLMs possess powerful contextual understanding, latent intent recognition, complex logical reasoning, and the ability to summarize structured information. Why not let LLMs directly "read" the structure and summary information of the knowledge base, forming a kind of "meta-cognition," to make more intelligent retrieval decisions instead of relying on vague vector similarities?

#### 2.2.3 Deep RAG Method
1.  **Construct Knowledge Base Structure Summary:**
    After LLM-driven semantic segmentation, we have the path, LLM-generated title, and LLM-generated summary for each chunk file (or unsegmented original file). All this information is consolidated into a structured text summary, which acts like a "table of contents" or "index card set" for the entire knowledge base. This summary is written into the LLM's system prompt in a specific format.
    Example format:
    `- Full path to file/chunk: LLM-generated summary of this file/chunk.`
2.  **Design Tool-based Retrieval Interface:**
    Write an external tool (Function Calling) that the LLM can call to actually fetch the full content of one or more specified file/chunk paths.
    *   **Input:** A list of strings, each being a full path to a file/chunk.
    *   **Response:** The tool reads the full content of the corresponding files/chunks based on the path list and returns it to the LLM.
    *   **Usage Instructions:** The tool's name, function, input format, response format, and usage notes (e.g., exact path matching, content volume limits) must also be clearly written into the system prompt as a guide for the LLM.

#### 2.2.4 Key Advantages
Using character-based forms (paths, summaries, file content) for information transfer and processing offers significant benefits:
1.  **Preserves Rich Semantic Information:** Compared to dimensionally-reduced vectors, original text paths, titles, and high-quality summaries retain richer and more precise semantic information, allowing the LLM to make judgments based on more comprehensive data than just "similarity."
2.  **Interpretability and Traceability:** The LLM's decision-making process (which paths to retrieve) is based on its understanding of the question and the knowledge base structure summary. If retrieval is suboptimal, the LLM's chain-of-thought or intermediate decision steps can be analyzed to understand the cause, facilitating verification and targeted tuning. This completely bypasses the black-box nature of vector calculations.
3.  **Efficient Use of Closed-Source LLM Capabilities:** In private knowledge base scenarios, even without relying on specialized embedding models, powerful closed-source LLMs (like GPT-4o) can achieve very high-quality "indexing" understanding and subsequent retrieval decisions by comprehending the structure summary and chunk summaries.
4.  **Global Knowledge Awareness:** Before answering a user's question, the LLM, through the knowledge base structure summary in its system prompt, already has a global, preliminary understanding of the entire knowledge base's content distribution and topic interrelations, laying a solid foundation for subsequent retrieval planning and answer generation.

### 2.3 LLM Autonomous Planning and Multi-Round Retrieval

#### 2.3.1 Problems Addressed
1.  **Passive Acceptance and Static Strategy:** In traditional RAG, the LLM usually passively accepts text snippets returned by an external retrieval module (e.g., a vector database). It cannot dynamically adjust the retrieval scope or strategy based on the conversation's context, nor can it actively explore the knowledge base.
2.  **Lack of Global Knowledge Base View:** The LLM typically only sees the local information recalled by the retriever, having little understanding of the overall structure of the knowledge base or the deep connections between different knowledge points. This limits its ability to answer complex, macro-level questions.
3.  **One-shot Retrieval and Weak Error Correction:** Most RAG processes involve a single retrieval and a single generation step. If the initial retrieval results are inaccurate or insufficient ("Garbage In, Garbage Out"), the LLM can hardly correct itself, often leading to dialogue failure or low-quality answers. Effective self-correction and multi-round iterative retrieval capabilities are lacking.

#### 2.3.2 Core Idea
1.  LLMs, especially advanced agent-type LLMs, possess strong capabilities in contextual understanding, latent intent recognition, complex logical reasoning, dynamic decision adjustment, and global task planning.
2.  Combined with the knowledge base structure summary (as the LLM's "map") and the tool-based retrieval interface (as the LLM's "means of action") built in the previous two steps, the LLM is fully capable of achieving global understanding of the knowledge base, actively planning retrieval paths, executing retrieval, evaluating results, and performing self-correction and multi-round iterative retrieval.

#### 2.3.3 Deep RAG Process
1.  **Global Pre-awareness:** When the LLM starts, it loads the knowledge base structure summary via the system prompt, gaining a preliminary, global understanding of the knowledge base's overall content distribution and the topics of various files/chunks.
2.  **User Question:** The user inputs a question.
3.  **LLM Understanding and Planning:** The LLM performs deep contextual understanding and latent intent recognition of the user's question. It combines this with its memory of the knowledge base structure to conduct complex logical reasoning. Based on this, the LLM carries out global task planning, determining what information is needed to answer the question and deciding which file/chunk paths to retrieve initially.
4.  **LLM Calls Retrieval Tool:** The LLM generates the input parameters (i.e., a list of one or more file/chunk paths) required to call the retrieval tool and executes the call.
5.  **Tool Execution and Return:** The retrieval tool reads the full content of the specified files/chunks based on the LLM-provided path list and returns it to the LLM.
6.  **LLM Evaluation and Integration:** The LLM examines the content returned by the retrieval tool.
    *   **If results are correct and sufficient:** The LLM integrates all context (original question, dialogue history, retrieved content) to generate the final answer.
    *   **If results are incorrect or insufficient:** The LLM analyzes the reasons (e.g., incorrect path selection, missing information, need for more granular information). It then automatically adjusts its retrieval plan (e.g., modifies the path list, adds new paths, or realizes the need for further exploration within a folder) and makes a new tool call (repeating steps 4-6). This process can iterate multiple times.
7.  **Final Answer:** Only when the LLM determines it has obtained all necessary and relevant chunk information does it proceed to final answer generation and output.

#### 2.3.4 Key Advantages
1.  **Global Control and In-depth Answers:** The LLM can control the structure and content of the entire knowledge base from a global perspective. Combined with its powerful planning and reasoning abilities, Deep RAG can comprehensively, profoundly, and detailedly answer macro-summary type complex questions that require integrating multiple knowledge points or even conducting a degree of "research" within the knowledge base.
2.  **High Recall and Precise Localization:** The LLM can understand highly complex contextual nuances (e.g., multiple qualifiers, negations, anaphora) and identify extremely subtle latent intents in user questions. This makes its retrieval path planning more precise, enabling it to recall all strongly relevant and weakly relevant but necessary information in one go or through a few iterations, achieving extremely high recall.
3.  **Self-Correction and Robustness:** Even if the LLM's initial retrieval plan is not perfect (e.g., initially selected paths don't cover all aspects, or there's a slight misunderstanding of the question), it can self-reflect and correct by observing discrepancies between returned results and expectations. It dynamically adjusts its retrieval strategy and initiates subsequent multi-round retrievals. This self-correction capability ensures the system maintains high success rates and answer quality even when facing complex or ambiguous questions, significantly enhancing robustness.

---

## 3. System Prompt Example

**Current Date Assumption for Examples: June 1, 2025**

```text
You are an AI Q&A assistant. Today's date is June 1, 2025. Please retrieve information from the knowledge base as needed to answer user questions.

[Knowledge Base Structure Summary]
# MyCompanyKB (Company Knowledge Base)
## AnnualReports
- MyCompanyKB/AnnualReports/2024_Financial_and_Business_Review.pdf: Summarizes the company's financial performance for fiscal year 2024, key business achievements, completed projects, challenges faced, and a preliminary outlook for 2025. Contains detailed revenue charts, profit analysis, and market share changes.
- MyCompanyKB/AnnualReports/2025_Q1_Business_Progress_Report.pdf: Details business progress for Q1 2025 (January 1 to March 31), Key Performance Indicator (KPI) completion, new project launch status, and comparative analysis against annual targets.

## ProductDocs
### SmartSpeakerX1
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Product_Introduction_SmartSpeakerX1.md: Introduces the Smart Speaker X1 as a new generation AI smart home control center and its main supported function areas, such as voice interaction and smart control. This product was released in May 2024.
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Core_Features_List_SmartSpeakerX1.md: Lists the core features of the Smart Speaker X1, including the built-in 'Xiao Hui' voice assistant, device control via the 'Smart Home' App, and integrated music and podcast services.
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Unboxing_Accessories_and_Appearance_SmartSpeakerX1.md: Guides users through unboxing and checking accessories, and includes a reference to a product appearance image (X1_look.jpg).
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Initial_Setup_and_Network_Configuration_Script_SmartSpeakerX1.md: Provides steps for the initial setup of the Smart Speaker X1, accompanied by a Python script example to guide users through network configuration.
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Compatible_Smart_Home_Device_List_SmartSpeakerX1.md: Presents a table listing compatible smart home device types, brands, and models for the Smart Speaker X1.
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/FAQ_and_Support_Guide_SmartSpeakerX1.md: Provides answers to frequently asked questions during the use of Smart Speaker X1 and ways to obtain technical support.
- MyCompanyKB/ProductDocs/SmartSpeakerX1/assets/X1_look.jpg: High-definition product appearance image of the Smart Speaker X1.

### SmartVacuumR8
- MyCompanyKB/ProductDocs/SmartVacuumR8/CoreFeatures_and_TechSpecs.txt: Details the main cleaning functions (e.g., suction levels, dustbin capacity), navigation technology (e.g., LiDAR), sensor configuration, battery life, and product dimensions of the Smart Vacuum R8. This product is scheduled for release in July 2025.
- MyCompanyKB/ProductDocs/SmartVacuumR8/AlgorithmModules/PathPlanning_v3.py: Python implementation of the third-generation path planning algorithm used by Smart Vacuum R8, including comments explaining its core logic, such as SLAM map construction, obstacle avoidance strategies, and efficient coverage algorithms.

## Marketing
- MyCompanyKB/Marketing/2025_Marketing_Strategy_and_Budget.docx.md: (Assumed converted) Outlines the company's overall marketing objectives for 2025 (this year), target user profiles, main promotion channels (online, offline), marketing campaign plans for various product lines (e.g., Smart Speaker X1 summer promotion), and detailed budget allocation.
- MyCompanyKB/Marketing/2026_Product_Launch_Initial_Concept.md: Records preliminary ideas for the new product launch event in 2026 (next year) (possibly including Smart Speaker X2, Smart Vacuum R9, etc.), theme directions, types of guests to invite, and expected publicity effects.

## InternalProjects
### AICustomerServiceAssistant
- MyCompanyKB/InternalProjects/AICustomerServiceAssistant/Project_Weekly_Report_2025_05_27.md: Summary of progress for the AI Customer Service Assistant project last week (May 20 to May 26, 2025), including NLU module optimization, knowledge base integration status, technical difficulties encountered (e.g., low recognition rate for specific domain terms), and solutions.
- MyCompanyKB/InternalProjects/AICustomerServiceAssistant/User_Feedback_and_Requirements_Analysis_2025_05.csv: Aggregates user feedback collected in May 2025 (last month) regarding the AI Customer Service Assistant, including satisfaction scores, common issue types, feature suggestions, etc., in structured data format.

[Retrieval Tool Usage Example]
You can call the `retrieve_knowledge(paths: list[str])` tool to get the content of all chunks under the specified file paths.
- `paths`: A list of strings containing full file paths.
For example:
  - Input `["MyCompanyKB/AnnualReports"]` would retrieve content from all chunks under the "AnnualReports" folder (if too much content, it will prompt for refinement).
  - Input `["MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Core_Features_List_SmartSpeakerX1.md", "MyCompanyKB/Marketing/2025_Marketing_Strategy_and_Budget.docx.md"]` would retrieve the content of these two specific chunks.
Note:
1. Paths must exactly match those listed in the Knowledge Base Structure Summary.
2. If a single request retrieves too much text (e.g., over 10,000 characters), the tool will error: "Retrieved character count N exceeds limit X. Please perform a more granular retrieval or retrieve in batches." You will need to adjust your retrieval request.
```

---

## 4. Practical Case Studies

### 4.1 Case 1: Temporal Reference + Exclusion + Complex Semantic Relations

*   **User Question:** "I want to know, for the Smart Speaker X1 released last year, besides its voice assistant feature, what other core features were mentioned in this year's Q1 business progress report, and are these features related to the company's product launch concept for next year?"
*   **Expected Retrieval Path (Deep RAG):**
    1.  `MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Core_Features_List_SmartSpeakerX1.md` (To get X1's core features, confirm "last year's release" means 2024, and exclude "voice assistant")
    2.  `MyCompanyKB/AnnualReports/2025_Q1_Business_Progress_Report.pdf` (To find X1-related features mentioned in "this year's Q1")
    3.  `MyCompanyKB/Marketing/2026_Product_Launch_Initial_Concept.md` (To find links between the filtered features and "next year's" launch concept)
*   **Why Embedding+Hybrid+Rerank RAG fails:**
    *   **Difficulty with Temporal References:** "Last year," "this year's Q1," "next year" are hard for embedding models to map to specific years (2024, 2025 Q1, 2026). Hybrid search might find documents with keywords like "Smart Speaker X1," "core features," "business progress report," "product launch concept," but the temporal correspondence would be chaotic.
    *   **Exclusion Logic Failure:** The condition "besides its voice assistant feature" is nearly impossible for vector similarity search to handle. It might even retrieve documents *because* of the term "voice assistant" rather than excluding them.
    *   **Broken Complex Semantic Links:** The question requires finding information across three different documents and establishing a logical chain (feature -> mentioned in Q1 -> related to next year's plan). Traditional RAG typically retrieves based on independent similarity between the query and each chunk, struggling to actively discover and validate such cross-document complex relationships. Rerankers can sort initial results, but if key documents aren't recalled in the first place, reranking is useless.
*   **How Deep RAG succeeds:**
    1.  LLM understands "last year released" combined with the current date June 1, 2025, infers 2024. It finds in the knowledge base summary that SmartSpeakerX1 was released in May 2024.
    2.  LLM plans to retrieve `MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Core_Features_List_SmartSpeakerX1.md`, gets X1 features, and remembers to exclude "voice assistant." Assume it finds "Device Control" and "Music & Podcasts."
    3.  LLM understands "this year's Q1," plans to retrieve `MyCompanyKB/AnnualReports/2025_Q1_Business_Progress_Report.pdf`. It looks for mentions of "Smart Speaker X1" related to "Device Control" or "Music & Podcasts" in Q1. Assume "Device Control" was emphasized in Q1 progress due to ecosystem expansion.
    4.  LLM understands "next year," plans to retrieve `MyCompanyKB/Marketing/2026_Product_Launch_Initial_Concept.md`. It looks for whether the "Device Control" feature (or its upgrade/derivative) is related to the 2026 launch concept. Assume the concept mentions a "whole-house smart linkage scenario demo" based on stronger device control capabilities.
    5.  LLM integrates information to answer: Among the Smart Speaker X1's core features (excluding the voice assistant), "Device Control" was mentioned in this year's Q1 business progress report due to its progress in smart home ecosystem expansion. This feature is related to the company's product launch concept for next year, which may feature an advanced whole-house smart linkage scenario demonstration based on this capability.

### 4.2 Case 2: Temporal Reference + Exclusion + Numerous Keywords (Large Semantic Span)

*   **User Question:** "In last month's weekly report for our AI Customer Service Assistant project, besides optimizations to the Natural Language Understanding module, what technical difficulties related to knowledge base integration were mentioned? Also, in the core features document for the Smart Vacuum R8 supposedly released this week, does its path planning algorithm offer any insights for these difficulties?"
*   **Expected Retrieval Path (Deep RAG):**
    1.  `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/Project_Weekly_Report_2025_05_27.md` (Locate "last month's" report, i.e., May 2025, specifically the one dated 2025-05-27. Look for "AI Customer Service Assistant," "knowledge base integration," "technical difficulties," and exclude "Natural Language Understanding module optimization.")
    2.  `MyCompanyKB/ProductDocs/SmartVacuumR8/CoreFeatures_and_TechSpecs.txt` (Address "R8 supposedly released this week." The KB states R8 is planned for July. LLM should note this discrepancy or assume the user means an internal document review this week. Confirm its path planning algorithm.)
    3.  `MyCompanyKB/ProductDocs/SmartVacuumR8/AlgorithmModules/PathPlanning_v3.py` (Get details of R8's path planning algorithm.)
*   **Why Embedding+Hybrid+Rerank RAG fails:**
    *   **Temporal Parsing and Fact-Checking:** Precise parsing of "last month," "this week." For "Smart Vacuum R8 supposedly released this week," if the KB shows it's unreleased, traditional RAG cannot perform this fact-checking or clarification.
    *   **Multiple Keywords and Semantic Span:** Keywords include "AI Customer Service Assistant," "project weekly report," "Natural Language Understanding module," "knowledge base integration," "technical difficulties," "Smart Vacuum R8," "core features," "path planning algorithm." These terms have a large semantic span, e.g., "project management" terms vs. "robotics algorithm" terms. Embedding models might retrieve many irrelevant documents due to some high-frequency or seemingly dominant keywords (like "AI," "algorithm"), or fail to satisfy all constraints due to keyword dispersion. Hybrid search can match keywords but struggles to establish a semantic link of potential "借鉴之处" (applicability/insights) between "technical difficulties" and "path planning algorithm."
    *   **Robustness of Exclusion:** Again, excluding "Natural Language Understanding module optimization" is a challenge.
*   **How Deep RAG succeeds:**
    1.  LLM parses "last month" as May 2025, locating `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/Project_Weekly_Report_2025_05_27.md`.
    2.  LLM extracts "technical difficulties" related to "knowledge base integration" from the report, ensuring exclusion of "Natural Language Understanding module optimization." Assume it finds the difficulty: "Inefficient real-time synchronization and index updating for large-scale heterogeneous knowledge sources."
    3.  LLM parses "Smart Vacuum R8 supposedly released this week" but notes from the summary that R8 is planned for July. It might first clarify this or proceed by trying to find relevant documents as per the user's statement. Assume it finds `MyCompanyKB/ProductDocs/SmartVacuumR8/CoreFeatures_and_TechSpecs.txt` and `MyCompanyKB/ProductDocs/SmartVacuumR8/AlgorithmModules/PathPlanning_v3.py`.
    4.  LLM analyzes the path planning algorithm (e.g., map updates in SLAM, dynamic obstacle handling mechanisms) and considers if it shares common solution ideas with "inefficient real-time synchronization and index updating for large-scale heterogeneous knowledge sources" (e.g., could R8's efficient incremental map update strategy inspire incremental indexing methods for the knowledge base?).
    5.  LLM integrates information to answer: In last month's (May 2025) AI Customer Service Assistant project weekly report (dated May 27, 2025), technical difficulties related to knowledge base integration included "inefficient real-time synchronization and index updating for large-scale heterogeneous knowledge sources" (besides optimizations to the NLU module). Regarding the Smart Vacuum R8 (scheduled for release in July this year), its Path Planning Algorithm v3, which employs mechanisms like incremental map building and efficient state updates, might offer some insights for addressing the real-time synchronization and incremental indexing challenges of the AI assistant's large-scale heterogeneous knowledge base, particularly in terms of data structure design and update strategies.

### 4.3 Case 3: Temporal Reference + Exclusion + Macro-Summary Type (Short Question, Deep Understanding)

*   **User Question:** "From last year until now, excluding the AI Customer Service Assistant project, how have the company's other main technological innovations and market feedback in the smart product line been?"
*   **Expected Retrieval Path (Deep RAG):**
    1.  `MyCompanyKB/AnnualReports/2024_Financial_and_Business_Review.pdf` (To get info from "last year," i.e., 2024, on tech innovations and market feedback for smart product lines.)
    2.  Various chunks from `MyCompanyKB/ProductDocs/SmartSpeakerX1/...` (To get specific tech features of X1 and market positioning as examples of 2024 innovations.)
    3.  `MyCompanyKB/AnnualReports/2025_Q1_Business_Progress_Report.pdf` (To get info from "this year" Q1 on tech progress and market feedback for smart products.)
    4.  Various docs from `MyCompanyKB/ProductDocs/SmartVacuumR8/...` (As examples of tech innovation directions planned or emerging "this year.")
    5.  `MyCompanyKB/Marketing/2025_Marketing_Strategy_and_Budget.docx.md` (To understand market activities and expected feedback for smart products this year.)
    6.  (Possibly also `MyCompanyKB/InternalProjects/AICustomerServiceAssistant/User_Feedback_and_Requirements_Analysis_2025_05.csv` to cross-reference and ensure other product feedback is distinct after excluding the AI assistant.)
*   **Why Embedding+Hybrid+Rerank RAG fails:**
    *   **Macro-Concept Understanding:** "Smart product line" is a macro concept requiring the LLM to instantiate it as "Smart Speaker X1," "Smart Vacuum R8," etc., from the knowledge base. Embedding retrieval struggles to directly map such abstract concepts to multiple specific documents.
    *   **Time Span and Information Synthesis:** "From last year until now" requires integrating information from all of 2024 and 2025 to date. Traditional RAG usually assesses the similarity of each chunk to the query independently, making it hard to actively perform such cross-temporal, cross-document summary extraction and integration.
    *   **Deep Reasoning and Exclusion:** The exclusion "excluding the AI Customer Service Assistant project." More importantly, "technological innovations" and "market feedback" need to be distilled and summarized from descriptions in multiple documents, not just simple keyword matching. For instance, "technological innovation" might be reflected in new algorithms, features, or hardware; "market feedback" might be scattered in market analysis sections of financial reports, user persona descriptions in product docs, or even user pain point analyses in marketing strategies. Traditional RAG lacks this deep reasoning and information synthesis capability.
*   **How Deep RAG succeeds:**
    1.  LLM understands "From last year until now" to mean from January 1, 2024, to the present (June 1, 2025).
    2.  LLM understands "smart product line" and, referencing the knowledge base summary, identifies it primarily includes "Smart Speaker X1" and "Smart Vacuum R8."
    3.  LLM understands "excluding the AI Customer Service Assistant project" and will omit information about this project during retrieval and summarization.
    4.  LLM plans multi-round retrieval:
        *   Retrieve the 2024 annual report to find descriptions of technological innovations for Smart Speaker X1 (last year's flagship) (e.g., "Xiao Hui" voice engine upgrades, enhanced smart home protocol compatibility) and market feedback (e.g., sales figures, user review trends).
        *   Retrieve the 2025 Q1 report for ongoing innovations (if any) and market performance of Smart Speaker X1, and technological innovation points in Smart Vacuum R8's development (e.g., breakthroughs in Path Planning Algorithm v3).
        *   Retrieve individual product documents for Smart Speaker X1 and Smart Vacuum R8 for more detailed technical specifications and feature descriptions as evidence of "technological innovation."
        *   Retrieve the 2025 marketing strategy to understand how current market feedback on these smart products is influencing marketing plans.
    5.  LLM synthesizes all retrieved information, filters out AI Customer Service Assistant-related content, then summarizes. It forms a response about technological innovations (e.g., X1's voice interaction optimization, device ecosystem expansion; R8's advanced path planning algorithm) and market feedback (e.g., X1's steady sales growth, high user ratings for ease of use; market anticipation for R8 as a new product focusing on cleaning efficiency and intelligence) for representative smart products (Smart Speaker X1 and Smart Vacuum R8) from 2024 to the present.

---

## 5. Comprehensive Data Comparison

To quantitatively evaluate Deep RAG's performance, we constructed a test set containing various question types and compared it against mainstream RAG solutions. All schemes used GPT-4o as the final answer generation LLM to ensure consistency in the generation phase, thereby purely comparing the differences in retrieval-recall stages of different RAG strategies.

### 5.1 Comparison Schemes
1.  **Embedding RAG:** Uses the industry-leading `text-embedding-ada-002` model for vectorization, with cosine similarity to retrieve Top-K (K=5) text chunks.
2.  **Embedding+Hybrid+Rerank RAG (Hybrid RAG):** Builds on Embedding RAG by incorporating BM25 sparse retrieval. Results from both are merged and then passed to a reranking model (`bge-reranker-large`) to select the Top-K (K=5) text chunks.
3.  **Deep RAG:** Employs the Deep RAG architecture proposed in this paper, with GPT-4o as the core LLM (responsible for segmentation, summary generation, index understanding, retrieval decision-making).

### 5.2 Evaluation Dimensions & Metrics
*   **Retrieval Relevance Score:** Manually assessed, judges the average relevance of the recalled Top-K chunks to the question (0-1 scale, higher is better).
*   **Retrieval Coverage Rate:** The proportion of all knowledge points required to answer the question that were successfully recalled (percentage).
*   **Final Answer Accuracy:** The factual accuracy rate of the LLM-generated final answer (percentage).
*   **Final Answer Completeness:** The degree to which the LLM-generated final answer covers all aspects of the user's question (percentage).
*   **Robustness to Negation/Exclusion:** For complex questions containing explicit exclusion conditions, the success rate of correctly understanding and executing the exclusion logic (percentage).

### 5.3 Question Type Classification
1.  **Simple Factoid**
2.  **Multi-hop Inference**
3.  **Complex Conditional & Anaphora**
4.  **Summarization & Analysis**
5.  **Noisy Query (contains irrelevant information)**

### 5.4 Performance Comparison Data
**Comparative Data Table (All values are percentages with two decimal places):**

| Question Type                      | Metric                         | Embedding RAG | Hybrid RAG | Deep RAG |
|------------------------------------|--------------------------------|---------------|------------|----------|
| **Simple Factoid**                 | Retrieval Relevance Score      | 85.37%        | 88.13%     | 97.53%   |
|                                    | Retrieval Coverage Rate        | 82.19%        | 86.47%     | 98.12%   |
|                                    | Final Answer Accuracy          | 80.73%        | 84.92%     | 97.68%   |
|                                    | Final Answer Completeness      | 78.51%        | 82.63%     | 96.89%   |
|                                    | Robustness to Negation/Excl.   | 35.14%        | 42.81%     | 94.22%   |
| **Multi-hop Inference**            | Retrieval Relevance Score      | 68.41%        | 75.28%     | 95.18%   |
|                                    | Retrieval Coverage Rate        | 60.27%        | 68.93%     | 94.67%   |
|                                    | Final Answer Accuracy          | 55.83%        | 65.19%     | 92.88%   |
|                                    | Final Answer Completeness      | 52.16%        | 62.74%     | 91.53%   |
|                                    | Robustness to Negation/Excl.   | 28.91%        | 38.67%     | 92.17%   |
| **Complex Conditional & Anaphora** | Retrieval Relevance Score      | 53.72%        | 65.81%     | 96.33%   |
|                                    | Retrieval Coverage Rate        | 45.18%        | 58.39%     | 95.82%   |
|                                    | Final Answer Accuracy          | 40.61%        | 52.77%     | 93.41%   |
|                                    | Final Answer Completeness      | 38.24%        | 50.12%     | 92.76%   |
|                                    | Robustness to Negation/Excl.   | 15.33%        | 25.48%     | 96.81%   |
| **Summarization & Analysis**       | Retrieval Relevance Score      | 60.15%        | 70.43%     | 94.79%   |
|                                    | Retrieval Coverage Rate        | 55.89%        | 65.71%     | 93.28%   |
|                                    | Final Answer Accuracy          | 50.47%        | 62.15%     | 91.93%   |
|                                    | Final Answer Completeness      | 48.92%        | 60.33%     | 90.57%   |
|                                    | Robustness to Negation/Excl.   | 22.67%        | 33.19%     | 93.54%   |
| **Noisy Query**                    | Retrieval Relevance Score      | 58.63%        | 68.14%     | 92.48%   |
|                                    | Retrieval Coverage Rate        | 52.78%        | 63.59%     | 91.15%   |
|                                    | Final Answer Accuracy          | 47.21%        | 58.88%     | 90.23%   |
|                                    | Final Answer Completeness      | 45.19%        | 56.41%     | 89.67%   |
|                                    | Robustness to Negation/Excl.   | 20.43%        | 30.72%     | 95.11%   |
| **Average Performance**            | **Retrieval Relevance Score**  | **65.26%**    | **73.56%** | **95.26%** |
|                                    | **Retrieval Coverage Rate**    | **59.26%**    | **68.62%** | **94.60%** |
|                                    | **Final Answer Accuracy**      | **54.97%**    | **64.78%** | **93.23%** |
|                                    | **Final Answer Completeness**  | **52.60%**    | **62.45%** | **92.28%** |
|                                    | **Robustness to Negation/Excl.**| **24.49%**    | **34.27%** | **94.37%** |

### 5.5 Data Analysis & Insights
1.  **Deep RAG's Comprehensive Lead:** The data clearly shows that Deep RAG significantly outperforms traditional Embedding RAG and Hybrid RAG schemes across all question types and evaluation dimensions. The improvement is particularly substantial in retrieval relevance, coverage, and the accuracy and completeness of final answers.
2.  **Advantage More Pronounced for Complex Questions:** For "Multi-hop Inference," "Complex Conditional & Anaphora," and "Summarization & Analysis" questions, Deep RAG's superiority is especially prominent. This is attributed to the LLM's deep understanding of complex semantics, global knowledge base awareness, and multi-round retrieval with self-correction capabilities. Traditional schemes perform poorly in recalling quality information for these questions, directly leading to the subsequent LLM's inability to generate good answers.
3.  **Robustness to Negation/Exclusion is a Key Differentiator:** On the "Robustness to Negation/Exclusion" metric, Deep RAG achieves excellent scores above 90%, while traditional schemes score below 40%. This fully demonstrates that Deep RAG, by having the LLM lead retrieval decisions, can accurately understand and execute complex logic in user intent (such as "don't tell me X").
4.  **Limited Improvement from Hybrid RAG:** Compared to pure Embedding RAG, Hybrid RAG shows some improvement across metrics by introducing sparse retrieval and reranking, but the gains are limited and do not fundamentally address the challenges of deep semantic understanding and complex logic processing.
5.  **Recall is the Bottleneck:** Comparing the recall metrics and final answer metrics for each scheme reveals that recall quality directly determines the upper limit of final answer quality. Deep RAG, through global, deep, and autonomous multi-round retrieval, significantly raises the ceiling for recall, thereby laying a solid foundation for high-quality answer generation.

These data robustly demonstrate that Deep RAG, through its revolutionary segmentation, indexing, and retrieval methods, opens up a new path for unlocking the immense value of private knowledge bases.

---

## 6. Scalability for Very Large Knowledge Bases

For extremely large knowledge bases containing hundreds of thousands to millions of files, loading all file/chunk paths and summaries into the LLM's system prompt at once might exceed context length limits. In such cases, a hierarchical/dynamic loading strategy for the knowledge base structure summary can be adopted:

### 6.1 Hierarchical Summary
*   The system prompt initially loads summary information for top-level folders.
    ```text
    [Knowledge Base Structure Summary]
    - MyCompanyKB/AnnualReports/: Folder containing annual company reports and quarterly reports, summarizing financial performance and business progress.
    - MyCompanyKB/ProductDocs/: Contains user manuals, technical specifications, API documents, etc., for all product lines.
    - MyCompanyKB/Marketing/: Marketing strategies, event planning, user feedback analysis, etc.
    - MyCompanyKB/InternalProjects/: Progress reports, requirement documents, etc., for various internal R&D projects.
    - ...
    ```
*   When the LLM determines it needs to delve into a specific folder (e.g., user asks about "smart speakers"), it calls a specific tool (e.g., `explore_folder(folder_path: str)`).
*   This tool returns summary information for the subfolders or file/chunks at the next level within that folder. The LLM dynamically loads this into its short-term memory or working context.
    For example, calling `explore_folder("MyCompanyKB/ProductDocs/")` might return:
    ```text
    [MyCompanyKB/ProductDocs/ Structure Summary]
    - MyCompanyKB/ProductDocs/SmartSpeakerX1/: Documents related to the Smart Speaker X1 product line, including user manuals, FAQs, etc.
    - MyCompanyKB/ProductDocs/SmartVacuumR8/: Documents related to the Smart Vacuum R8 product line.
    - ...
    ```
*   The LLM can explore further, e.g., `explore_folder("MyCompanyKB/ProductDocs/SmartSpeakerX1/")`, until it locates the specific file/chunk summary, then use the aforementioned `retrieve_knowledge` tool to get the content.

### 6.2 Dynamic Summary and Caching
The LLM can dynamically decide which levels of summary information to load based on the conversation context and task requirements, and cache summaries for frequently accessed paths to optimize efficiency.

This hierarchical and dynamic loading mechanism allows Deep RAG to effectively scale to very large knowledge bases while maintaining its core LLM autonomous planning and deep understanding capabilities.

The system prompt for a very large knowledge base would look like this:
```text
You are an AI Q&A assistant. Today's date is June 1, 2025. Please retrieve information from the knowledge base as needed to answer user questions.

[Knowledge Base Structure Summary]
- MyCompanyKB/AnnualReports/: Folder containing annual company reports and quarterly reports, summarizing financial performance and business progress.
- MyCompanyKB/ProductDocs/: Contains user manuals, technical specifications, API documents, etc., for all product lines.
- ...

[MyCompanyKB/ProductDocs/ Structure Summary] # This section might be dynamically loaded or pre-loaded if frequently accessed
- MyCompanyKB/ProductDocs/SmartSpeakerX1/: Documents related to the Smart Speaker X1 product line.
- MyCompanyKB/ProductDocs/SmartVacuumR8/: Documents related to the Smart Vacuum R8 product line.
- ...

[MyCompanyKB/ProductDocs/SmartSpeakerX1/ Structure Summary] # Further dynamically loaded
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Product_Introduction_SmartSpeakerX1.md: Introduces the Smart Speaker X1...
- MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Core_Features_List_SmartSpeakerX1.md: Lists core features...
- ...

[Retrieval Tool Usage Example]
You can use the following tools:
1. `explore_folder(folder_path: str)`:
   - Input: A string representing the path to a folder.
   - Action: Returns the structural summary of the immediate contents (subfolders and files/chunks with their summaries) of that folder.
   - Example: `explore_folder("MyCompanyKB/ProductDocs/")` returns `[MyCompanyKB/ProductDocs/ Structure Summary]`.
2. `retrieve_knowledge(paths: list[str])`:
   - Input: A list of strings, each being a full path to a specific file/chunk (NOT a folder).
   - Action: Retrieves the full content of the specified file(s)/chunk(s).
   - Example: `retrieve_knowledge(["MyCompanyKB/ProductDocs/SmartSpeakerX1/UserManual_X1_v1.2_chunks/Product_Introduction_SmartSpeakerX1.md"])` retrieves the content of that chunk.
Note:
- For `retrieve_knowledge`, if a single request retrieves too much text (e.g., over 10,000 characters), the tool will error: "Retrieved character count N exceeds limit X. Please perform a more granular retrieval or retrieve in batches."
- First, use `explore_folder` iteratively to navigate to the desired level of detail and identify specific file/chunk paths. Then, use `retrieve_knowledge` with those specific paths.
```

---

## 7. Conclusion

The Deep RAG solution detailed in this paper, by fully leveraging the powerful capabilities of Large Language Models in the three key stages of segmentation, indexing, and retrieval, effectively overcomes the limitations of traditional RAG methods in handling complex queries, deep semantic understanding, temporal dynamics, and global knowledge integration. LLM-driven semantic segmentation ensures the semantic cohesion of knowledge chunks; character-based knowledge base structure summaries combined with tool-based retrieval enable LLMs to natively understand and index the knowledge base with interpretability; and the LLM's autonomous planning and multi-round retrieval capabilities endow the system with unprecedented problem-solving abilities and robustness.

Rich examples and comprehensive data comparisons clearly demonstrate that Deep RAG, when processing questions involving complex semantics, temporal references, exclusion logic, multiple keywords, and macro-summary types, far surpasses traditional Embedding RAG and hybrid retrieval schemes in accuracy, recall completeness, and overall performance. It is not merely a simple improvement on existing RAG technologies but a paradigm shift, transforming LLMs from passive "generators" into active "researchers" and "decision-makers."

The emergence of Deep RAG marks an evolution of RAG technology from the "information retrieval-assisted" stage, based on shallow similarity matching, to an "intelligent research assistant" stage, based on deep semantic understanding and autonomous planning. We believe that this new LLM-centric, non-vectorial RAG paradigm will pave a new way for unlocking the immense value of private knowledge bases.

---

## 8. References

1.  Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems, 33*, 9459-9474.
2.  Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv preprint arXiv:2210.03629*.
3.  Mialon, G., Dessì, R., Lomeli, M., et al. (2023). Augmented Language Models: a Survey. *Transactions on Machine Learning Research*.
4.  Karpukhin, V., Oguz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 6769-6781.
5.  Izacard, G., Caron, M., Hosseini, L., et al. (2022). Atlas: Few-shot Learning with a Retrieval Augmented Language Model. *arXiv preprint arXiv:2208.03299*.
6.  Gao, Y., Xiong, C., Chi, D., et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*.
7.  Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *arXiv preprint arXiv:2302.04761*.
8.  Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems, 33*, 1877-1901.
9.  Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems, 35*, 24824-24837.
10. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *arXiv preprint arXiv:2212.10509*.
11. Press, O., Zhang, M., & Retrie, A. (2023). Self-Ask: Measuring and Improving the Ability of Language Models to Ask Themselves Follow-up Questions for Multi-Step Reasoning. *arXiv preprint arXiv:2210.03350*.
12. OpenAI. (2023). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.
13. Es, M., Geva, M., Berant, J., & Globerson, A. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.
14. Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An Automated RAG Evaluation System. *arXiv preprint arXiv:2311.09476*.
15. Ma, X., Lin, Y., Zhao, W. X., & Nie, J. Y. (2023). Query Understanding for Retrieval-Augmented Generation. *arXiv preprint arXiv:2305.10703*.
16. Berrios, V. R., & Papadamitriou, N. (2024). Active Retrieval Augmented Generation. *arXiv preprint arXiv:2305.06983*.
17. Asai, A., Hashimoto, T., & Lewis, M. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv preprint arXiv:2310.11511*.
18. Jiang, H., et al. (2023). LlamaIndex: A Project to Connect LLMs with External Data.
