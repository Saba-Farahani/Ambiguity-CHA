"""
LLM integration utilities for STYLE implementation.
Handles GPT-3.5 integration for user simulation and clarification question generation.
"""

import os
import json
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
from ..config import Config


class LLMIntegration:
    def __init__(self, config: Config):
        """
        Initialize LLM integration.

        Args:
            config: Configuration object containing API key and other settings
        """
        print("Current working directory:", os.getcwd())

        # Load .env file only once
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            print(f"Loading .env file from: {env_path}")
            load_dotenv(env_path, override=True)
        else:
            print("Warning: No .env file found in the current directory")

        # Get API key from config or environment
        self.api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided in config or environment variables"
            )

        print("Using API key:", self.api_key[:8] + "..." + self.api_key[-4:])

        openai.api_key = self.api_key
        self.model = config.MODEL_NAME  # Use MODEL_NAME from config
        self.config = config

    def generate_clarification_question(
        self,
        query_history: List[str],
        retrieved_docs: List[str],
        retrieval_scores: List[float],
    ) -> str:
        """
        Generate a clarification question using few-shot CoT prompting.

        Args:
            query_history: List of previous queries in the conversation
            retrieved_docs: List of retrieved documents
            retrieval_scores: List of retrieval scores for the documents

        Returns:
            Generated clarification question
        """
        # Prepare few-shot examples
        examples = [
            {
                "query": "What movies are playing?",
                "docs": [
                    "Avengers: Endgame is showing",
                    "The Lion King is a new release",
                ],
                "question": "Are you interested in action movies or family-friendly films?",
            },
            {
                "query": "Where should I eat?",
                "docs": ["Italian restaurant downtown", "Sushi place near the beach"],
                "question": "Do you prefer Italian cuisine or Japanese food?",
            },
        ]

        # Prepare the prompt
        prompt = "Generate a clarification question based on the user's query and retrieved documents.\n\n"
        prompt += "Examples:\n"
        for ex in examples:
            prompt += f"Query: {ex['query']}\n"
            prompt += f"Documents: {', '.join(ex['docs'])}\n"
            prompt += f"Clarification Question: {ex['question']}\n\n"

        prompt += "Current case:\n"
        prompt += f"Query: {query_history[-1]}\n"
        prompt += f"Documents: {', '.join(retrieved_docs)}\n"
        prompt += "Clarification Question:"

        # Call GPT-3.5
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates clarification questions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=100,
        )

        return response.choices[0].message.content.strip()

    def simulate_user_response(self, user_intent: str, question: str) -> str:
        """
        Simulate a user's response based on intent and question.

        Args:
            user_intent: The user's true intent
            question: The question to respond to

        Returns:
            Simulated user response
        """
        try:
            prompt = f"""User Intent: {user_intent}
Question: {question}

Generate a natural user response that aligns with the intent:"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating user response: {e}")
            # Fallback to using intent directly
            return user_intent

    def evaluate_clarification_quality(
        self, query: str, clarification_question: str, user_intent: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a clarification question.

        Args:
            query: Original user query
            clarification_question: Generated clarification question
            user_intent: User's true intent

        Returns:
            Dictionary containing quality metrics
        """
        prompt = f"""Evaluate the quality of this clarification question.
        
Original Query: {query}
Clarification Question: {clarification_question}
User's Intent: {user_intent}

Rate the following aspects from 0.0 to 1.0:
1. Helpfulness: Does the question help narrow down the user's intent?
2. Relevance: Is the question relevant to the user's query?
3. Specificity: Is the question specific enough to get useful information?

Provide the ratings in JSON format."""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating the quality of clarification questions.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        try:
            ratings = json.loads(response.choices[0].message.content)
            return ratings
        except json.JSONDecodeError:
            return {"helpfulness": 0.0, "relevance": 0.0, "specificity": 0.0}

    def generate_answer(
        self, domain: str, intent: str, query_history: List[str]
    ) -> str:
        """Generate a final answer based on the conversation history."""
        prompt = f"""Domain: {domain}
User Intent: {intent}
Query History: {query_history}

Generate a comprehensive answer that addresses the user's intent:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def check_sufficient_info(
        self, query: str, retrieved_docs: List[str], intent: str
    ) -> bool:
        """Check if we have sufficient information to answer the query."""
        prompt = f"""Query: {query}
Retrieved Documents: {retrieved_docs}
User Intent: {intent}

Do we have sufficient information to provide a complete answer? Answer with 'yes' or 'no':"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3,
            )
            return "yes" in response.choices[0].message.content.lower()
        except Exception as e:
            print(f"Error checking sufficient info: {e}")
            return False

    def generate_clarification(
        self, domain: str, intent: str, query_history: List[str]
    ) -> str:
        """Generate a clarifying question to better understand the user's needs."""
        prompt = f"""Domain: {domain}
User Intent: {intent}
Query History: {query_history}

Generate a clarifying question to better understand the user's needs:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return ""

    def evaluate_answer(self, answer: str, intent: str) -> bool:
        """Evaluate if the answer satisfies the user's intent."""
        prompt = f"""Answer: {answer}
User Intent: {intent}

Does this answer fully satisfy the user's intent? Answer with 'yes' or 'no':"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3,
            )
            return "yes" in response.choices[0].message.content.lower()
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False

    def generate_query(self, user_intent: str, query_history: List[str]) -> str:
        """Generate a query based on user intent and conversation history."""
        prompt = f"""Given the user's intent and conversation history, generate a relevant query.
        
User Intent: {user_intent}
Conversation History: {query_history}

Generate a query that will help gather information to address the user's intent:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates relevant queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating query: {e}")
            return "What information do you need?"

    def generate_response(self, query: str) -> str:
        """Generate a response to a query."""
        prompt = f"""Given the following query, generate a helpful response:
        
Query: {query}

Response:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides informative responses.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response at this time."

    def generate_answer(
        self, user_intent: str, query_history: List[str], retrieved_docs: List[str]
    ) -> str:
        """Generate a final answer based on the conversation history and retrieved documents."""
        prompt = f"""Given the user's intent, conversation history, and retrieved documents, generate a comprehensive answer.
        
User Intent: {user_intent}
Conversation History: {query_history}
Retrieved Documents: {retrieved_docs}

Generate a detailed answer that addresses the user's intent:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides comprehensive answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I'm sorry, I couldn't generate an answer at this time."

    def check_sufficient_info(
        self, user_intent: str, query_history: List[str], retrieved_docs: List[str]
    ) -> bool:
        """Check if we have sufficient information to answer the user's intent."""
        prompt = f"""Given the user's intent, conversation history, and retrieved documents, determine if we have sufficient information to provide a satisfactory answer.
        
User Intent: {user_intent}
Conversation History: {query_history}
Retrieved Documents: {retrieved_docs}

Do we have enough information to provide a satisfactory answer? (yes/no):"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that evaluates information sufficiency.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip().lower()
            return "yes" in answer
        except Exception as e:
            print(f"Error checking information sufficiency: {e}")
            return False

    def generate_clarification(
        self, user_intent: str, query_history: List[str], retrieved_docs: List[str]
    ) -> str:
        """Generate a clarifying question based on the current context."""
        prompt = f"""Given the user's intent, conversation history, and retrieved documents, generate a clarifying question to better understand the user's needs.
        
User Intent: {user_intent}
Conversation History: {query_history}
Retrieved Documents: {retrieved_docs}

Generate a clarifying question:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that asks clarifying questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return (
                "Could you please provide more details about what you're looking for?"
            )

    def evaluate_answer(
        self, user_intent: str, answer: str, retrieved_docs: List[str]
    ) -> bool:
        """Evaluate if the answer satisfies the user's intent."""
        prompt = f"""Given the user's intent, the generated answer, and retrieved documents, evaluate if the answer is satisfactory.
        
User Intent: {user_intent}
Answer: {answer}
Retrieved Documents: {retrieved_docs}

Is this answer satisfactory? (yes/no):"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that evaluates answer quality.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=10,
            )
            evaluation = response.choices[0].message.content.strip().lower()
            return "yes" in evaluation
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False

    def generate_text(self, text: str, task: str = "paraphrase") -> str:
        """
        Generate text based on the input text and task.

        Args:
            text: Input text to process
            task: Task to perform (e.g., "paraphrase", "summarize")

        Returns:
            Generated text
        """
        try:
            if task == "paraphrase":
                prompt = f"""Paraphrase the following text while maintaining its meaning:
                
{text}

Paraphrased version:"""
            else:
                prompt = f"""Process the following text for {task}:
                
{text}

Processed version:"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant that performs {task} tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            # Return original text as fallback
            return text
