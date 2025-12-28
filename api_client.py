"""
Client to interact with Liquid AI API server
"""

import requests
import json
from typing import List, Dict, Any


class LiquidAIClient:
    """Client for Liquid AI LFM2 model API"""
    
    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def health(self) -> Dict[str, Any]:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Generate completion for a prompt"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
        }
        
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=self.headers,
            json=payload,
        )
        
        result = response.json()
        return result["choices"][0]["text"]
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Generate chat completion"""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


def main():
    """Example usage"""
    client = LiquidAIClient()
    
    print("Checking server health...")
    try:
        health = client.health()
        print(f"Server status: {json.dumps(health, indent=2)}\n")
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        print("Make sure the server is running: docker-compose up")
        return
    
    # Example 1: Simple completion
    print("=" * 80)
    print("Example 1: Simple Completion")
    print("=" * 80)
    prompt = "What is C. elegans?"
    print(f"Prompt: {prompt}")
    try:
        response = client.completion(prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Chat completion
    print("=" * 80)
    print("Example 2: Chat Completion")
    print("=" * 80)
    messages = [
        {"role": "user", "content": "Say hi in JSON format"}
    ]
    print(f"Messages: {messages}")
    try:
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Another chat
    print("=" * 80)
    print("Example 3: Complex Chat")
    print("=" * 80)
    messages = [
        {"role": "user", "content": "Explain artificial intelligence in 2 sentences"}
    ]
    print(f"Messages: {messages}")
    try:
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()