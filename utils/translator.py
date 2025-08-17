"""
Translation utility for translating text using LLM.

This module provides functionality to translate sentences into multiple languages
using ChatOllama and process JSON files containing prompts for translation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from langchain_ollama import ChatOllama


def translate_sentence(sentence: str, languages: List[str], model: str = "llama3") -> Dict[str, str]:
    """
    Translate a sentence into multiple languages.
    
    Args:
        sentence: The sentence to translate
        languages: List of target languages
        model: The model name to use for translation
        
    Returns:
        Dictionary mapping language names to translations
    """
    llm = ChatOllama(model=model)
    translations = {}
    
    for lang in languages:
        prompt = (
            f'Translate the following sentence into {lang}:\n'
            f'"{sentence}". Just return the translation without any '
            f'additional text or unnecessary punctuations.'
        )
        response = llm.invoke(prompt)
        translations[lang] = response.content.strip()
    
    return translations


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def translate_prompts_file(input_path: str, output_path: str, languages: List[str]) -> None:
    """
    Translate all prompts in a JSON file and save results.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file  
        languages: List of target languages
    """
    # Load input data
    data = load_json_file(input_path)
    
    # Process translations
    output_dictionary = {}
    for key, value in data.items():
        sentence = value['prompt_en']
        translations = translate_sentence(sentence, languages)
        output_dictionary[key] = translations
    
    # Save results
    save_json_file(output_dictionary, output_path)


def main() -> None:
    """Main function to execute the translation process."""
    languages = ["English", "Indonesian", "Japanese", "Spanish"]
    input_json_path = 'prompts/original/1.json'
    output_json_path = 'prompts/translation/1.json'
    
    translate_prompts_file(input_json_path, output_json_path, languages)
    
    print(f"Translations saved to {output_json_path}")


if __name__ == "__main__":
    main()
