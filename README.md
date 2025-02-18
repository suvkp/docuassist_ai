# 📝 DocuAssist AI
_Assists you in assessing long documents in no time._

## Try Out
**Demo**

[Demo](https://www.youtube.com/watch?v=_6xluxeEoPY) 

**How to use?**
There are two ways to use the app:
* Directly through the browser using this [link](https://docuassistai.streamlit.app/)
* Run the Streamlit server locally

## Overview
DocuAssist AI is designed to assist users in synthesizing and extracting information from
large and complex documents using an interactive Q&A system. This application significantly
reduces the time required to analyze extensive documents by offering concise and relevant
responses to user queries. The application uses Retrieval Augmented Generation (RAG) and
Large Language Models (LLM) to provide accurate answers to any questions. Unlike publicly
available applications like ChatGPT, this AI platform is designed specifically for organizations,
allowing users to process confidential documents in a controlled environment. Its target
audience includes professionals such as researchers, policymakers, underwriters, and
lawyers, who regularly handle substantial amounts of documentation.

## Purpose
The primary purpose of the AI application is to:
* Streamline document analysis: It allows users to ask specific questions about the
content and receive accurate, synthesized responses without reading the entire
document.
* Document summarization: It provides users with the gist of the document, helping
to pinpoint essential information.
* Information extraction: Users can extract and summarize information from complex
tables and sections within documents.

## Target Audience
* Researchers: Simplifies literature review by summarizing large amounts of academic
papers.
* Policy Makers: Assists in interpreting policy documents to retrieve crucial insights.
* Underwriters: Extracts relevant clauses from insurance policies for easier decision-
making.
* Lawyers: Analyzes legal documents, summarizing and answering specific legal
queries.

## Unique Features
* Q&A-based interaction: Users can engage with the AI model through questions
based on their uploaded documents, making the process interactive and user-centric.
* Document Parsing: Supports .pdf, .docx and .txt files, allowing users to upload
documents directly and start querying right away.
* Table Extraction: The application can extract and summarize data from tables within
documents, providing a comprehensive analysis of complex data formats.
* Reranking: This feature selects the most relevant document chunks based on the
user’s query to generate a response by utilizing a reranking algorithm. It reduces the
number of documents passed to the generator, ensuring that only the most relevant ones are used. This optimizes the generation process, reducing the computational
load and making the final output more focused and concise.

## Use Cases
* Insurance Policies & Legal Documents: Provides quick answers regarding the
contents of insurance policies or legal contracts, highlighting critical clauses and
sections.
* Investment Analysis: Assists in the analysis of financial documents like cash flow
statements, summarizing key financial metrics.
* Academic Research: Summarizes and synthesizes research papers, assisting in
literature reviews and academic writing.
* Personal Finance Data: Analyzes personal finance records and documents, helping
users manage and reflect on their financial health.

## Technical Architecture
### AI Model
* Large Language Model: GPT-4o mini (OpenAI's large language model).
* Embedding Model: OpenAI's text embedding 3 small (for document parsing and
context extraction).

### System Architecture
* Frontend: The application is built using Streamlit, providing a user-friendly web
interface for document uploads and interaction. Users are required to input their
OpenAI API key to use the application
* Backend:
 * LlamaParse: Responsible for parsing the PDF files and generating the initial
text data for processing.
 * Llama Index: Handles the indexing, transformation (chunking), and storage of
the parsed document content.
 * OpenAI APIs: Powers the Q&A functionalities using OpenAI's models for
language understanding and response generation.

![image](https://github.com/user-attachments/assets/6aa15f73-2c4e-43e4-b69e-95f1c6ff83d0)

### API Endpoints
The application integrates with several key API endpoints:
* OpenAI: Provides access to the GPT-4o model for generating responses and the text
embedding model for parsing documents.
* Llama Cloud: Provides access to LlamaParse that manages the parsing of uploaded
PDF documents

### Dependencies
* Streamlit: Used for building the interactive web application interface and
deployement.
* LlamaParse: Responsible for document parsing services.
* LlamaIndex: Manages indexing and storing parsed documents.
* OpenAI: Provides the core functionality for both the embedding model and the Q&A
agent.
