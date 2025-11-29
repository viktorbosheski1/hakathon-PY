import os
import json
from typing import Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMHelper:
    """Helper class for Azure OpenAI LLM operations"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.model_name = os.getenv("REASONING_MODEL_NAME", "gpt-4")
        self.api_version = os.getenv("REASONING_API_VERSION", "2024-12-01-preview")
        
        # Validate required environment variables
        if not self.azure_endpoint:
            raise ValueError("AZURE_ENDPOINT is not set in environment variables")
        if not self.azure_api_key:
            raise ValueError("AZURE_API_KEY is not set in environment variables")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            api_version=self.api_version
        )
        
        print(f"LLM Helper initialized with model: {self.model_name}")
    
    def read_prompt_from_file(self, file_path: str) -> str:
        """
        Read prompt content from a file
        
        Args:
            file_path: Path to the prompt file
            
        Returns:
            Content of the prompt file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading prompt file {file_path}: {str(e)}")
    
    def format_user_prompt(self, prompt_template: str, **kwargs) -> str:
        """
        Format user prompt with variables
        
        Args:
            prompt_template: Template string with placeholders
            **kwargs: Variables to replace in the template
            
        Returns:
            Formatted prompt string
        """
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise Exception(f"Missing variable in prompt template: {str(e)}")
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Call Azure OpenAI LLM and return response
        
        Args:
            system_prompt: System prompt content
            user_prompt: User prompt content
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 1000)
            response_format: Expected response format ("json" or "text")
            
        Returns:
            Dictionary with LLM response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Prepare common parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            # Check if model supports max_tokens or max_completion_tokens
            # gpt-5-mini and newer models use max_completion_tokens
            if "gpt-5" in self.model_name.lower() or "o1" in self.model_name.lower():
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
            
            # Add response format if JSON
            if response_format == "json":
                params["response_format"] = {"type": "json_object"}
            
            # Call Azure OpenAI API
            response = self.client.chat.completions.create(**params)
            
            # Extract the response content
            content = response.choices[0].message.content.strip()
            
            # Parse JSON if expected
            if response_format == "json":
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, try to extract JSON from the response
                    # Sometimes the model adds extra text around the JSON
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise Exception(f"Failed to parse JSON response: {str(e)}")
            else:
                return {"content": content}
                
        except Exception as e:
            raise Exception(f"Error calling LLM: {str(e)}")
    
    def get_internal_documents_answer(
        self,
        question: str,
        answer_qa: str,
        relevant_document: str,
        system_prompt_path: str = "./prompts/internal_docs_system.txt",
        user_prompt_path: str = "./prompts/internal_docs_user.txt"
    ) -> Dict[str, Any]:
        """
        Get answer from LLM based on internal documents
        
        Args:
            question: User's question
            answer_qa: Answer from Q&A database
            relevant_document: Most relevant document from internal documents
            system_prompt_path: Path to system prompt file
            user_prompt_path: Path to user prompt template file
            
        Returns:
            Dictionary with 'answer' and 'score' keys
        """
        try:
            # Read prompts from files
            system_prompt = self.read_prompt_from_file(system_prompt_path)
            user_prompt_template = self.read_prompt_from_file(user_prompt_path)
            
            # Format user prompt with variables
            user_prompt = self.format_user_prompt(
                user_prompt_template,
                question=question,
                answer_qa=answer_qa,
                relevant_document=relevant_document
            )
            
            # Call LLM
            response = self.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=1000,
                response_format="json"
            )
            
            # Validate response has required keys
            if "answer" not in response or "score" not in response:
                raise Exception("LLM response missing required keys 'answer' or 'score'")
            
            return response
            
        except Exception as e:
            print(f"Error getting internal documents answer: {e}")
            return {"answer": "", "score": 0.0}
    
    def get_combined_answer(
        self,
        question: str,
        answer_qa: str,
        answer_internal_documents: str,
        system_prompt_path: str = "./prompts/combined_system.txt",
        user_prompt_path: str = "./prompts/combined_user.txt"
    ) -> Dict[str, Any]:
        """
        Get combined answer from LLM based on both Q&A and internal documents answers
        
        Args:
            question: User's question
            answer_qa: Answer from Q&A database
            answer_internal_documents: Answer from internal documents
            system_prompt_path: Path to system prompt file
            user_prompt_path: Path to user prompt template file
            
        Returns:
            Dictionary with 'answer' and 'score' keys
        """
        try:
            # Read prompts from files
            system_prompt = self.read_prompt_from_file(system_prompt_path)
            user_prompt_template = self.read_prompt_from_file(user_prompt_path)
            
            # Format user prompt with variables
            user_prompt = self.format_user_prompt(
                user_prompt_template,
                question=question,
                answer_qa=answer_qa,
                answer_internal_documents=answer_internal_documents
            )
            
            # Call LLM
            response = self.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=1000,
                response_format="json"
            )
            
            # Validate response has required keys
            if "answer" not in response or "score" not in response:
                raise Exception("LLM response missing required keys 'answer' or 'score'")
            
            return response
            
        except Exception as e:
            print(f"Error getting combined answer: {e}")
            return {"answer": "", "score": 0.0}


# Initialize global LLM helper instance
try:
    llm_helper = LLMHelper()
except Exception as e:
    print(f"Warning: Failed to initialize LLM Helper: {e}")
    llm_helper = None
