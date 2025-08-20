import re
import json
import os
import sys
import random
import pandas as pd
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv


def setup_environment(env_directory) -> None:
    """Initialize environment variables from .env file."""
    # Get the path relative to the script location
    env_path = Path(__file__).parent.parent / env_directory / '.env'
    if env_path.exists():
        print(f"Found .env file at {env_path}, loading environment variables...")
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
        
    # Verify API key is available
    if not os.getenv("GRAPHRAG_API_KEY"):
        print("Error: GRAPHRAG_API_KEY environment variable is not set")
        sys.exit(1)


def load_file_content(file_path: str, is_json: bool = False) -> Any:
    """Load content from a file with error handling."""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            if is_json:
                return json.load(f)
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        sys.exit(1)


def run_eval(query_file: str, result1_file: str, result2_file: str, output_file_path: str, answer_mapping_path: str):
    """Evaluate pairs of answers for given questions using LLM-as-a-judge."""
    # Check if output directories exist
    output_dir = os.path.dirname(output_file_path)
    mapping_dir = os.path.dirname(answer_mapping_path)
    
    if not os.path.exists(output_dir):
        print(f"Error: evaluation_results output directory does not exist: {output_dir}")
        sys.exit(1)
    
    if not os.path.exists(mapping_dir):
        print(f"Error: Answer mapping directory does not exist: {mapping_dir}")
        sys.exit(1)

    client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))

    # Load and parse query file
    data = load_file_content(query_file)
    queries = re.findall(r"- Question \d+: (.+)", data)
    if not queries:
        print(f"Error: No questions found in {query_file}")
        sys.exit(1)

    # Load and parse result files
    answers1 = load_file_content(result1_file, is_json=True)
    answers2 = load_file_content(result2_file, is_json=True)
    filename1 = os.path.splitext(os.path.basename(result1_file))[0]
    filename2 = os.path.splitext(os.path.basename(result2_file))[0]
    
    try:
        answers1 = [i["result"] for i in answers1]
        answers2 = [i["result"] for i in answers2]
    except (KeyError, TypeError):
        print("Error: Invalid format in result files. Expected list of objects with 'result' key")
        sys.exit(1)

    if len(answers1) != len(queries) or len(answers2) != len(queries):
        print(f"Error: Mismatch in number of questions ({len(queries)}) and answers ({len(answers1)}, {len(answers2)})")
        sys.exit(1)

    results = []
    # Create a list to track which file was used for which answer
    answer_mapping = []
    
    for i, (query, ans1, ans2) in enumerate(zip(queries, answers1, answers2)):
        # Randomly decide which answer goes to which position
        if random.choice([True, False]):
            answer1, answer2 = ans1, ans2
            file_mapping = {"Answer 1": filename1, "Answer 2": filename2}
        else:
            answer1, answer2 = ans2, ans1
            file_mapping = {"Answer 1": filename2, "Answer 2": filename1}
            
        # Store the mapping
        answer_mapping.append({
            "Question #": f"Question {i+1}",
            "question": query,
            "Answer 1": file_mapping["Answer 1"],
            "Answer 2": file_mapping["Answer 2"]
        })

        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format, ensuring that all output is valid JSON, and that all string values within the JSON are properly escaped to adhere to JSON standards:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        print(f"Processing evaluation {i+1}/{len(queries)}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            result = {
                "request_id": f"request-{i+1}",
                "question": query,
                "evaluation": response.choices[0].message.content
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing evaluation {i+1}: {str(e)}")
            # Save partial results before exiting
            if results:
                print("Saving partial results...")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                mapping_df = pd.DataFrame(answer_mapping)
                mapping_df.to_csv(answer_mapping_path, index=False)
                print(f"Partial evaluation results written to {output_file_path}")
                print(f"Answer mapping written to {answer_mapping_path}")
            sys.exit(1)

    # Save evaluation results to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create and save the answer mapping dataframe
    mapping_df = pd.DataFrame(answer_mapping)
    mapping_df.to_csv(answer_mapping_path, index=False)
    print(f"Evaluation results written to {output_file_path}")
    print(f"Answer mapping written to {answer_mapping_path}")


if __name__ == "__main__":
    setup_environment('_MRP') # Main directory of GraphRAG run: .env file with OpenAI API key should be in this directory
    run_eval(query_file="_MRP_eval/query_file.txt",                              # File containing queries that were answered by the two runs
             result1_file="_MRP_eval/default/result_file_default_swap_5%.json",  # File containing results from the swap 5% run (or another run for comparison)
             result2_file="_MRP_eval/default/result_file_default_run_2.json",    # File containing results from the default run (or another run for comparison)
             output_file_path="_MRP_eval/evaluation_results_swap_40%.json",      # Output file path for evaluation results
             answer_mapping_path="_MRP_eval/answer_mapping_swap_40%.csv")        # File to save the mapping of result file answers that were shuffled
