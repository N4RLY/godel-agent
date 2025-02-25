import os
import logging
from agent import GodelAgent
import time
import ast
import openai
from dotenv import load_dotenv
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_code(code: str) -> bool:
    """Validate the syntax of the provided code."""
    try:
        logger.info("Validating code syntax...")
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Code validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during code validation: {e}")
        return False

def clean_code(code: str) -> str:
    """Clean and extract Python code from GPT response."""
    try:
        logger.info("Cleaning code...")
        return code[code.find('```python')+9:code.rfind('```')].strip()
    except Exception as e:
        logger.error(f"Code cleaning failed: {e}")
        return ""

def call_gpt4(prompt: str) -> str:
    """Call GPT-4 API with the given prompt."""
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT")
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to improve code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return ""

def get_current_code() -> str:
    logger.info("Inspecting source code.")
    try:
        return inspect.getsource(GodelAgent)
    except Exception as e:
        logger.error(f"Failed to get source code: {str(e)}")
    return ""

def get_improvement_prompt(code: str, goal: str) -> str:
    """Generate the prompt for code improvement."""
    return f"""
    Given the following:
    Current code: {code}
    Improvement goal: {goal}

    If you think that the code is already satisfactory, please respond with the same code.
    Otherwise, suggest improvements to the code while maintaining its core functionality.
    Return ONLY the complete updated class definition without any markdown formatting or explanatory text.
    The response should contain the entire class definition directly with 'class GodelAgent:' and contain only Python code.
    For example:
    ```python
    class GodelAgent:
        def __init__(self):
            pass
        def solve(self):
            pass
    ```
    """

def save_modified_code_to_file(code: str, version: float) -> None:
    """Save the modified code to a file with versioning."""
    filename = f"agent.py"
    try:
        with open(filename, 'w') as f:
            f.write(code)
        logger.info(f"Modified code saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save modified code: {e}")

def modify_code(code: str) -> str:
    # Modify the agent's code with the provided new code
    try:
        namespace = {}
        exec(code, namespace)
        new_agent_class = namespace.get('GodelAgent')
        if not new_agent_class:
            raise ValueError("New class definition not found in the provided code.")
        
        new_instance = new_agent_class()
        new_instance.history = agent.history.copy()
        new_instance.version = round(agent.version + 0.1, 2)
        new_instance.modifications = agent.modifications + [time.time()]
        
        agent.__class__ = new_instance.__class__
        agent.__dict__.update(new_instance.__dict__)
        
        agent.history.append({
            'timestamp': time.time(),
            'version': agent.version,
            'new_code': code
        })
        logger.info(f"Successfully updated to version {agent.version}")
        time.sleep(1)  # Add a small delay to prevent too rapid modifications

    except Exception as e:
        logger.error(f"Self-modification failed: {e}")

def reflect(agent: GodelAgent) -> dict:
    """Return a reflection of the agent's current state."""
    return {
        'version': agent.version,
        'num_modifications': len(agent.modifications),
        'history': agent.history
    }

def fix_code_with_llm(failed_code: str, goal: str) -> str:
    """Use LLM to suggest fixes for the provided code that failed validation."""
    prompt = f"""
    The following code failed validation:
    {failed_code}
    
    Please suggest improvements to fix the code while maintaining its core functionality.
    The improvement goal is: {goal}
    
    Return ONLY the complete updated class definition without any markdown formatting or explanatory text.
    The response should contain the entire class definition directly with 'class GodelAgent:' and contain only Python code.
    For example:
    ```python
    class GodelAgent:
        def __init__(self):
            pass
        def solve(self):
            pass
    ```
    """
    return call_gpt4(prompt)


goal = "Return the result of exp^10"
answer = ''
max_depth = 2

if __name__ == "__main__":
    # Initialize the agent
    agent = GodelAgent()
    logger.info("Starting the improvement process...")
    
    # Execute the improvement loop
    try:
        # Start recursive improvement
        depth = 0
        while depth < max_depth:
            logger.info(f"Starting recursive improvement at depth {depth}...")
            
            # Get current state
            current_code = get_current_code()
            # Generate and process improvement
            prompt = get_improvement_prompt(current_code, goal)
            suggested_code = call_gpt4(prompt)
            if suggested_code == current_code:
                logger.info("No changes needed. Exiting the improvement process.")
                break
            logger.info(f"Suggested code: {suggested_code}")
            code = clean_code(suggested_code)
            logger.info(f"Cleaned code: {code}")
            
            retries = 0
            while retries < 2:
                if validate_code(code):
                    modify_code(code)
                    save_modified_code_to_file(code, agent.version)
                    if answer and agent.solve()==answer:
                        depth = max_depth
                    else:
                        depth += 1
                    break
                else:
                    logger.warning(f"Code validation failed at depth {depth}")
                    logger.info("Attempting to fix the code using LLM...")
                    code = clean_code(fix_code_with_llm(code, goal))
                    logger.info(f"Cleaned code: {code}")
                    retries += 1

        logger.info("Final state:")
        logger.info(reflect(agent))
        logger.info(f"Final solution: {agent.solve()}")
    except Exception as e:
        logger.error(f"An error occurred during the improvement process: {e}")