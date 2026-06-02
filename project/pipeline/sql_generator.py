import os
import sys
from dotenv import load_dotenv

# Load explicitly from project/.env
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(PROJECT_DIR, ".env")
load_dotenv(env_path)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google import genai
from google.genai import types

# Initialize the new Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open(os.path.join(os.path.dirname(__file__), "system_prompt.txt"), "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

fk_path = os.path.join(os.path.dirname(__file__), "fk_resolution.txt")
if os.path.exists(fk_path):
    with open(fk_path, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT += f"\n\n## FOREIGN KEY RESOLUTION RULES\n{f.read()}"

def build_user_prompt(
    user_question: str,
    table_chunks: list[str],
    rule_chunks: list[str],
    sample_chunks: list[str]
) -> str:
    """
    Input:
        user_question  — raw natural language question from user
        table_chunks   — top-k retrieved table chunks (strings)
        rule_chunks    — top-k retrieved business rule chunks (strings)
        sample_chunks  — top-k retrieved sample query chunks (strings)

    Output:
        fully assembled user prompt string to send to Gemini
    """

    parts = []

    # ── schema context ────────────────────────────────────────────
    if table_chunks:
        schema_block = "\n\n".join(table_chunks)
        parts.append(f"RELEVANT TABLES:\n{schema_block}")

    # ── business rules ────────────────────────────────────────────
    if rule_chunks:
        rules_block = "\n\n".join(rule_chunks)
        parts.append(f"RELEVANT RULES:\n{rules_block}")

    # ── sample queries ────────────────────────────────────────────
    if sample_chunks:
        samples_block = "\n\n".join(sample_chunks)
        parts.append(f"SAMPLE QUERIES:\n{samples_block}")

    # ── user question ─────────────────────────────────────────────
    parts.append(f"QUESTION:\n{user_question}")

    return "\n\n".join(parts)


def generate_sql(user_query: str, retrieved_chunks: dict) -> str:
    user_prompt = build_user_prompt(
        user_question=user_query,
        table_chunks=retrieved_chunks.get("tables", []),
        rule_chunks=retrieved_chunks.get("rules", []),
        sample_chunks=retrieved_chunks.get("queries", [])
    )

    # Use the updated client.models.generate_content pattern
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=6000
        )
    )
    
    # Handle response text directly
    sql = response.text.strip() if response.text else ""
    
    # Clean up formatting if it outputs code blocks
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        sql = "\n".join(lines).strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
            
    return sql

if __name__ == "__main__":
    print("This file contains the core SQL generation logic.")
    print("To run the full pipeline dashboard, use: streamlit run project/main.py")
