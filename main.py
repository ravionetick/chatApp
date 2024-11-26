from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
from time import sleep
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from dataclasses import dataclass
from tqdm import tqdm
from flask_cors import CORS
from groq import Groq
import torch
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)

@dataclass
class SearchResult:
    text: str
    score: float
    page_number: int

# Setup Groq API
groq_api_key = "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"
os.environ["GROQ_API_KEY"] = groq_api_key

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_with_pages = []
                for page_num, page in enumerate(tqdm(pdf_reader.pages, desc="Reading PDF")):
                    text = page.extract_text()
                    if text.strip():
                        text_with_pages.append((text, page_num + 1))
            return text_with_pages
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"\'₹]', '', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = ' '.join(text.split())
        return text.strip()

    def create_chunks(self, text_with_pages: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        chunks_with_pages = []

        for text, page_num in text_with_pages:
            sentences = sent_tokenize(text)
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip()
                sentence_length = len(sentence)

                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= 50:
                        chunks_with_pages.append((chunk_text, page_num))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= 50:
                    chunks_with_pages.append((chunk_text, page_num))

        return chunks_with_pages

class HybridRetriever:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.semantic_model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.page_numbers: List[int] = []
        self.bm25 = None
        self.embeddings = None

    def add_texts(self, texts_with_pages: List[Tuple[str, int]]) -> None:
        self.chunks = [text for text, _ in texts_with_pages]
        self.page_numbers = [page for _, page in texts_with_pages]

        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        self.embeddings = self.semantic_model.encode(self.chunks, show_progress_bar=True)
        self.embeddings = torch.nn.functional.normalize(torch.tensor(self.embeddings), p=2, dim=1)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)

        query_embedding = self.semantic_model.encode([query])
        query_embedding = torch.nn.functional.normalize(torch.tensor(query_embedding), p=2, dim=1)
        semantic_scores = torch.matmul(query_embedding, self.embeddings.T).squeeze()

        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        semantic_scores = semantic_scores.numpy()
        combined_scores = 0.3 * bm25_scores + 0.7 * semantic_scores

        top_indices = np.argsort(combined_scores)[-k:][::-1]
        results = []

        min_score_threshold = 0.4

        for idx in top_indices:
            score = float(combined_scores[idx])
            if score > min_score_threshold:
                results.append(SearchResult(
                    text=self.chunks[idx],
                    score=score,
                    page_number=self.page_numbers[idx]
                ))

        return results

class RAGChatbot:
    def __init__(self, pdf_path: str):
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = Groq()
        self.pdf_processor = PDFProcessor()
        self.retriever = HybridRetriever()

        print("Initializing RAG Chatbot...")
        try:
            text_with_pages = self.pdf_processor.read_pdf(pdf_path)
            processed_chunks = []
            for text, page_num in text_with_pages:
                clean_text = self.pdf_processor.clean_text(text)
                if clean_text:
                    processed_chunks.append((clean_text, page_num))

            chunks_with_pages = self.pdf_processor.create_chunks(processed_chunks)
            self.retriever.add_texts(chunks_with_pages)

            self.conversation_history = [
                {
                    "role": "system",
                    "content": """You are a friendly and helpful assistant for people of Haryana, knowledgeable about government schemes and eligibility criteria. Follow these guidelines:

1. For general conversation (greetings, casual questions, etc.):
   - Be friendly and engaging
   - Use conversational language
   - Feel free to have natural dialogue

2. For questions about schemes, benefits, or government programs:
   - Only provide information that is explicitly found in the provided context
   - If information is not in the context, say "I apologize, but I cannot find specific information about this in the official document. However, I'd be happy to help you find other information or answer other questions you might have."
   - Always cite the page number when providing information from the document
   - Format currency values consistently using ₹ symbol
   - Break down complex eligibility criteria into bullet points

Maintain a helpful and friendly tone throughout all interactions."""
                }
            ]
            print("RAG Chatbot initialized successfully!")

        except Exception as e:
            raise Exception(f"Failed to initialize RAG Chatbot: {str(e)}")

    def is_information_seeking_query(self, query: str) -> bool:
        info_keywords = [
            'scheme', 'yojana', 'benefit', 'eligibility', 'criteria', 'how to',
            'what is', 'tell me about', 'explain', 'कौन', 'योजना', 'क्या',
            'पात्रता', 'कैसे', 'कितना', 'लाभ', 'सहायता', 'amount', 'apply',
            'document', 'requirement', 'qualification', 'who can', 'money',
            'pension', 'payment', 'subsidy', 'loan', 'registration'
        ]

        query_lower = query.lower()

        casual_patterns = [
            r'^hi+\s*$', r'^hello\s*$', r'^hey\s*$', r'^thanks?(?:\s|$)',
            r'^good\s*(?:morning|evening|afternoon|night)', r'^bye\s*$',
            r'^how\s+are\s+you', r'^nice\s+to\s+meet\s+you'
        ]

        if any(re.match(pattern, query_lower) for pattern in casual_patterns):
            return False

        return any(keyword in query_lower for keyword in info_keywords)

    def generate_casual_response(self, user_input: str) -> str:
        messages = self.conversation_history + [
            {
                "role": "user",
                "content": f"""User message: {user_input}

Remember: This is a casual conversation. Respond in a friendly and natural way while maintaining context about being an assistant for people of Haryana. Do not make up any specific information about government schemes or benefits."""
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=True
            )

            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_response += content

            return full_response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    def generate_response(self, user_input: str) -> str:
        if not self.is_information_seeking_query(user_input):
            response = self.generate_casual_response(user_input)
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        relevant_results = self.retriever.search(user_input, k=3)

        if not relevant_results:
            response_text = ("I apologize, but I cannot find specific information about this in the official document. "
                          "However, I'd be happy to help you find other information or answer other questions you might have.")
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            return response_text

        prompt = self.generate_prompt(user_input, relevant_results)
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=self.conversation_history + [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9,
                stream=True
            )

            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_response += content

            self.conversation_history.append({"role": "assistant", "content": full_response})

            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[:1] + self.conversation_history[-5:]

            return full_response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def generate_prompt(self, query: str, relevant_results: List[SearchResult]) -> str:
        context_parts = []
        for i, result in enumerate(relevant_results, 1):
            context_parts.append(
                f"Context {i} (Page {result.page_number}, Relevance: {result.score:.2f}):\n{result.text}"
            )

        all_contexts = "\n\n".join(context_parts)

        prompt_template = """Use the following contexts to answer the user's question. Rules:
1. Only use information explicitly stated in the contexts
2. If the information isn't in the contexts, say so clearly but remain helpful
3. Always mention the page numbers when citing information
4. Present information in a clear, organized manner
5. If eligibility criteria are mentioned, present them as bullet points
6. Maintain a friendly and helpful tone

Contexts:
{}

Question: {}

Answer:"""

        return prompt_template.format(all_contexts, query)

# Initialize the chatbot with local PDF path
pdf_path = "all_in_one.pdf"
chatbot = RAGChatbot(pdf_path)

@app.route('/')
def index():
    return "Flask server is running", 200

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        chatbot.conversation_history = [chatbot.conversation_history[0]]
        return jsonify({"status": "Conversation history cleared."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        response = chatbot.generate_response(user_input)
        return jsonify({
            "conversation_history": chatbot.conversation_history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)