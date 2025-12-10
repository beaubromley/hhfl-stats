"""
LLM Integration for SQL generation and result interpretation
Supports: Google Gemini, OpenAI, and Manual mode
"""
import requests
import json


def get_schema_context():
    """Return database schema context for LLM"""
    return """
DATABASE SCHEMA REFERENCE:

Tables:
1. players (player_id, player_name, code)
2. games (game_id, game_date, season, game_number, game_suffix, game_code, 
          captain1, captain2, mvp, team1_score, team2_score, overtime, is_tie, week_number)
3. player_game_stats (stat_id, game_id, player_id, captain,
          completions, attempts, completion_ratio, passing_yards, passing_tds, interceptions_thrown,
          qbr, qb_avg_drive, rush_attempts, rush_yards, rush_tds,
          receptions, receiving_yards, receiving_tds,
          tackles, sacks, interceptions, int_tds, fumble_recoveries, fumble_tds,
          kickoff_tds, safeties, win_loss, offensive_fantasy, defensive_fantasy, total_fantasy,
          player_game_label, streak, game_count)

Key Relationships:
- player_game_stats.player_id -> players.player_id
- player_game_stats.game_id -> games.game_id

Common Patterns:
- Get player stats: JOIN player_game_stats pgs with players p ON pgs.player_id = p.player_id
- Get game info: JOIN player_game_stats pgs with games g ON pgs.game_id = g.game_id
- Career totals: Use SUM() and GROUP BY player_id
- Season stats: Filter WHERE g.season = X
- Win rate: SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END)
"""


# ============================================================================
# GOOGLE GEMINI
# ============================================================================

def generate_sql_gemini(question, api_key, model="gemini-2.5-flash"):
    """Generate SQL using Google Gemini API via REST"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        schema = get_schema_context()
        
        prompt = f"""You are a SQL expert. Generate a valid SQLite query to answer this question about football statistics.

{schema}

User Question: {question}

CRITICAL INSTRUCTIONS:
- ALWAYS use LIMIT 10 or more (even if question asks for "the most" or "who has")
- Return top results so we can see context and compare
- Examples:
  * "Who has the most X?" → Return TOP 10 by X
  * "What is the highest X?" → Return TOP 10 by X
  * "Show me the best X" → Return TOP 10-20 by X
- Return ONLY the SQL query
- No explanations, no markdown formatting, no code blocks
- Include offsetting data to support the analysis (ex. if asked for passing yards, include passing TDs, completion percentages, etc.)
- Use proper JOINs between tables
- Round decimals to 1 decimal place with ROUND()
- Handle NULL values with NULLIF or COALESCE
- For player names, always JOIN with the players table
- Always ORDER BY the key metric in descending order

SQL Query:"""
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2000,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_msg = error_json['error'].get('message', error_msg)
            except:
                pass
            return None, f"API Error {response.status_code}: {error_msg}"
        
        result = response.json()
        
        try:
            candidate = result['candidates'][0]
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            
            if finish_reason not in ['STOP', 'MAX_TOKENS']:
                return None, f"Response blocked. Reason: {finish_reason}"
            
            sql = candidate['content']['parts'][0]['text'].strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            if sql.startswith('"') and sql.endswith('"'):
                sql = sql[1:-1]
            if sql.startswith("'") and sql.endswith("'"):
                sql = sql[1:-1]
            
            return sql, None
            
        except (KeyError, IndexError) as e:
            return None, f"Error parsing response: {e}"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def interpret_results_gemini(question, sql, results_df, api_key, model="gemini-2.5-flash"):
    """Interpret query results using Google Gemini API via REST"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        # Include more context - show top results
        results_text = results_df.head(20).to_string(index=False) if not results_df.empty else "No results found"
        
        prompt = f"""You are analyzing football statistics query results.

Original Question: {question}

SQL Query Used:
{sql}

Query Results (showing top results for context):
{results_text}

IMPORTANT:
- Answer the ORIGINAL question specifically
- If asked "who has the most", identify the #1 player but also mention the context (e.g., "Jimmy Quinn leads with X, followed by Troy Fite with Y")
- Include specific numbers and player names
- Be conversational and enthusiastic
- Add interesting thoughts or insights that may address a side question or follow up question to the original
- Provide interesting insights or comparisons from the top results

Answer:"""
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000},
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        
        if response.status_code != 200:
            return None, f"API Error {response.status_code}"
        
        result = response.json()
        
        try:
            interpretation = result['candidates'][0]['content']['parts'][0]['text'].strip()
            return interpretation, None
        except:
            return None, "Error parsing response"
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# OPENAI
# ============================================================================

def generate_sql_openai(question, api_key, model="gpt-3.5-turbo"):
    """Generate SQL using OpenAI API"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        
        schema = get_schema_context()
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a SQL expert. Generate valid SQLite queries.

{schema}

CRITICAL: ALWAYS use LIMIT 10 or more, even for "who has the most" questions.
Return top results for context.

Return ONLY the SQL query, no explanations."""
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return None, f"OpenAI Error {response.status_code}: {response.text}"
        
        result = response.json()
        sql = result['choices'][0]['message']['content'].strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        return sql, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def interpret_results_openai(question, sql, results_df, api_key, model="gpt-3.5-turbo"):
    """Interpret query results using OpenAI"""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        
        results_text = results_df.head(20).to_string(index=False) if not results_df.empty else "No results found"
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are analyzing football statistics. Answer the specific question asked, but provide context from the top results shown."
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}

Results (top results for context):
{results_text}

Answer the original question specifically, then provide interesting context from the other top results."""
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return None, f"OpenAI Error {response.status_code}"
        
        result = response.json()
        interpretation = result['choices'][0]['message']['content'].strip()
        
        return interpretation, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def test_api_key(api_key, model="gemini-2.5-flash", provider="gemini"):
    """Test if the API key is valid"""
    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": "Say 'Working'"}]}]}
            response = requests.post(url, json=payload, timeout=10)
        
        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'Working'"}],
                "max_tokens": 10
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "API key is valid!"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"
