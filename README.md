# AI Resume Screener

This project is a Resume Matching system that uses RAG (Retrieval-Augmented Generation) to process job descriptions and resumes, and then match them based on user queries.

## Features

- Upload and process job descriptions and resumes
- Match resumes to job descriptions using advanced NLP techniques
- Provide detailed analysis and ranking of candidates
- User-friendly web interface

## Project Workflow 

![diagram-export-9-23-2024-11_03_38-PM](https://github.com/user-attachments/assets/3246e9b8-816c-492d-9ff0-dcbbb01e0b6e)



## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- NLP: LangChain, Google's Generative AI
- Vector Store: FAISS
- File Processing: PyPDF2, python-docx

## Project Structure
```bash
AI_resume_screener/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── models.py
│ ├── rag_operations.py
│ └── utils.py
├── frontend/
│ └── streamlit_app.py
├── faiss_index/
├── static/
├── requirements.txt
├── .env
└── README.md
```


## Setup

1. Clone the repository:
```bash 
git clone https://github.com/yourusername/AI_resume_screener.git
cd AI_resume_screener
```


2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```


4. Set up your .env file:
- Copy the .env.example file to .env
- Replace 'your_google_api_key_here' with your actual Google API key

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run frontend/streamlit_app.py
```


3. Open your web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501)

## Usage

1. Upload a job description PDF
2. Upload one or more resume PDFs
3. Click "Process Documents" to analyze the uploaded files
4. Enter a query in the text box (e.g., "Find the best candidate for a senior software engineer position")
5. Click "Match Resumes" to see the analysis and rankings

## API Endpoints

- `POST /process_documents`: Upload and process job description and resumes
- `POST /match_resumes`: Match resumes based on a query

For detailed API documentation, run the backend and visit `http://localhost:8000/docs`


## Acknowledgments

- Thanks to the creators and maintainers of FastAPI, Streamlit, LangChain, and FAISS
- Google's Generative AI team for providing powerful language models

## Important Note :

The project still has some bugs and is still under development.


## Future Scope & Improvements : 

Remove all the bugs and add more features.

Here are some improvements which I wish to consider in future for pre-processing, chunking, retrieval, and matching:

1. Pre-processing improvements:

- Implement more advanced HTML tag and URL removal techniques using libraries like BeautifulSoup for better cleaning of web-sourced documents [6].
- Expand contracted words to their full forms to improve semantic understanding [6].
- Add emoji and emoticon handling to better process social media or informal text data [6].
- Implement spell checking to correct common errors in resumes and job descriptions [6].
- Use Parts of Speech (POS) tagging to provide additional context for words in documents [7][8].

2. Chunking improvements:

- Implement semantic chunking instead of fixed-size chunking to create more meaningful segments based on topic coherence [3].
- Use a combination of sentence splitting and recursive chunking to create chunks that balance semantic coherence and size constraints [3].
- Experiment with different chunk sizes and overlap percentages to find the optimal balance for your specific use case [1][3].

3. Retrieval improvements:

- Implement a hybrid search approach that combines keyword-based (e.g., BM25) and semantic similarity searches for more comprehensive results [2].
- Use Maximal Marginal Relevance (MMR) to diversify search results and reduce redundancy [2].
- Implement advanced filtering options to allow users to narrow down search results based on specific criteria like date, author, or document sections [4].

4. Matching improvements:

- Fine-tune your embedding model on domain-specific data (e.g., resumes and job descriptions) to improve semantic understanding in your specific context [2].
- Implement a re-ranking step after initial retrieval to further refine the relevance of matched documents [5].
- Incorporate user feedback and interaction data to continuously improve the matching algorithm over time [4][5].

5. Overall system improvements:

- Implement an efficient indexing system using inverted indexes to speed up document retrieval [5].
- Use advanced data structures and algorithms to optimize search and retrieval speed, especially for large datasets [5].
- Implement a user feedback loop to gather information on the relevance of matches and use this data to improve the system over time [4][5].
- Implement Better RAG with DSPy framework for prompt augmentation and with chainofThought prompting : https://dspy-docs.vercel.app/docs/tutorials/rag
- Implement a Faq section to sort down frequently asked questions. This Q&A will be saved in database. It can reduce the number of calls to LLM and optimize costs.
- Implement auto-correction and auto-completion for better user experience.
- Implement knowledge base and Graph RAG for better retrieval. 
- Implement sub LLMs for routing LLMs for different types of query from the recruiter. 
- Perform more comprehensive EDA on resume dataset - for better chunking, retrieval and generation based on our specific dataset.

By implementing these improvements, you can enhance the accuracy, efficiency, and user experience of your resume matching system. Remember to test each improvement individually to measure its impact on your specific use case.

Citations:
[1] https://www.rackspace.com/blog/how-chunking-strategies-work-nlp
[2] https://www.restack.io/p/information-retrieval-knowledge-nlp-methods-cat-ai
[3] https://www.pinecone.io/learn/chunking-strategies/
[4] https://teamhub.com/blog/improving-document-search-and-retrieval-through-technology/
[5] https://www.pickl.ai/blog/information-retrieval-in-nlp/
[6] https://www.einfochips.com/blog/nlp-text-preprocessing/
[7] https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/
[8] https://exchange.scale.com/public/blogs/preprocessing-techniques-in-nlp-a-guide
