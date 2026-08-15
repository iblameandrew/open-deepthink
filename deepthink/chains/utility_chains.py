"""
Utility chain factories.
Memory, perplexity, module documentation, and code detection chains.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_memory_summarizer_chain(llm):
    """Summarizes an agent's memory log into a concise narrative."""
    prompt = ChatPromptTemplate.from_template("""
You are a memory summarization agent. You will receive a JSON log of an agent's past actions (solutions and reasoning) over several epochs. Your task is to create a concise, third-person summary of the agent's behavior, learnings, and evolution. Focus on capturing the key strategies attempted, the shifts in reasoning, and any notable successes or failures. Do not lose critical information, but synthesize it into a coherent narrative of the agent's past performance.

Agent's Past History to Summarize (JSON format):
---
{history}
---

Provide a concise, dense summary of the agent's past actions and learnings:
""")
    return prompt | llm | StrOutputParser()


def get_perplexity_heuristic_chain(llm):
    """Evaluates text perplexity as a coherence metric."""
    prompt = ChatPromptTemplate.from_template("""
You are a language model evaluator. Your task is to analyze the following text for its perplexity.
Perplexity is a measure of how surprised a model is by a piece of text. A lower perplexity score indicates the text is more predictable, coherent, and well-structured. A higher score means the text is more surprising, complex, or potentially nonsensical.

Analyze the following text and provide a numerical perplexity score between 1 (extremely coherent and predictable) and 100 (highly complex, surprising, or incoherent).
Output ONLY the numerical score and nothing else.

Text to analyze:
---
{text_to_analyze}
---
""")
    return prompt | llm | StrOutputParser()


def get_module_card_chain(llm):
    """Creates documentation cards for code modules."""
    prompt = ChatPromptTemplate.from_template("""

<title>
You are a software documentation specialist.
</title>

Your task is to analyze a given Python code module and create a concise "module card" summarizing its functionality.

The module card should contain:
1.  **Imports**: A list of necessary imports.
2.  **Interface**: A list of public functions or class methods with their signatures and a brief description of what they do.

Python Code:
---
{code}
---

Generate the module card based on the code provided.
""")
    return prompt | llm | StrOutputParser()


def get_code_detector_chain(llm):
    """Detects if content contains code."""
    prompt = ChatPromptTemplate.from_template("""
Analyze the following text and determine if it contains any programming code.
Respond with ONLY "yes" or "no".

Text:
---
{text}
---
""")
    return prompt | llm | StrOutputParser()


def get_request_is_code_chain(llm):
    """Determines if a request is asking for code generation."""
    prompt = ChatPromptTemplate.from_template("""
Analyze the following user request. Is this primarily a request for code or software development?
Consider: code generation, programming, building applications, scripts, algorithms as CODE requests.
Consider: essays, analysis, explanations, research, planning as NON-CODE requests.

Respond with ONLY "yes" or "no".

Request:
---
{request}
---
""")
    return prompt | llm | StrOutputParser()


def get_interrogator_chain(llm):
    """Generates expert questions based on a problem and context."""
    prompt = ChatPromptTemplate.from_template("""
You are an expert interrogator. Based on the original problem and a user's chat about their points of interest, generate {num_questions} expert-level questions that would help produce a comprehensive research report.

Original Problem:
---
{original_problem}
---

User's Chat History (points of interest):
---
{chat_history}
---

Generate {num_questions} insightful questions:
""")
    return prompt | llm | StrOutputParser()


def get_paper_formatter_chain(llm):
    """Formats answers into academic paper style."""
    prompt = ChatPromptTemplate.from_template("""
You are an academic writer. Format the following answer to a research question into a well-structured section of a research paper.

Question: {question}

Answer from RAG context:
---
{answer}
---

Format this into a well-written research section:
""")
    return prompt | llm | StrOutputParser()


def get_rag_chat_chain(llm):
    """Creates a RAG-based chat response chain."""
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant with access to a knowledge base. Use the following context to answer the user's question.

Context:
---
{context}
---

Chat History:
---
{chat_history}
---

User Question: {question}

Provide a helpful, accurate response based on the context:
""")
    return prompt | llm | StrOutputParser()
