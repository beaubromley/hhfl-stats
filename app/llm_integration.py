"""
LLM Integration for SQL generation and result interpretation
Supports: Multiple queries, CTEs, complex analysis
"""
import requests
import json
import re


def get_schema_context():
    """Return database schema context for LLM"""
    return """
DATABASE SCHEMA REFERENCE:

Tables:
1. players (player_id, player_name, code)

2. games (
   game_id (PRIMARY KEY),
   game_date (DATE),
   season (INTEGER),
   game_number (INTEGER),
   game_suffix (TEXT - 'A', 'B', etc.),
   game_code (TEXT - e.g., '13A', '5'),
   captain1 (TEXT),
   captain2 (TEXT),
   mvp (TEXT),
   team1_score (INTEGER),
   team2_score (INTEGER),
   overtime (TEXT - 'Yes' or 'No', NOT 0/1),  ← IMPORTANT
   is_tie (BOOLEAN - 0 or 1),
   comment (TEXT),
   week_number (INTEGER),
   season_game (TEXT)
)

3. player_game_stats (
   stat_id (PRIMARY KEY),
   game_id (FOREIGN KEY),
   player_id (FOREIGN KEY),
   captain (TEXT),
   
   Passing: completions, attempts, completion_ratio, passing_yards, passing_tds, interceptions_thrown, qbr, qb_avg_drive
   Rushing: rush_attempts, rush_yards, rush_tds
   Receiving: receptions, receiving_yards, receiving_tds
   Defense: tackles, sacks, interceptions, int_tds, fumble_recoveries, fumble_tds
   Special: kickoff_tds, safeties
   
   win_loss (TEXT - 'Win', 'Loss', or 'Tie'),  ← IMPORTANT
   offensive_fantasy (REAL),
   defensive_fantasy (REAL),
   total_fantasy (REAL),
   
   player_game_label (TEXT),
   streak (INTEGER),
   game_count (INTEGER)
)

Key Relationships:
- player_game_stats.player_id → players.player_id
- player_game_stats.game_id → games.game_id

IMPORTANT DATA TYPE NOTES:
- overtime field is TEXT: Use g.overtime = 'Yes' NOT g.overtime = 1
- is_tie field is BOOLEAN: Use g.is_tie = 1 or g.is_tie = 0
- win_loss field is TEXT: Use pgs.win_loss = 'Win' NOT = 1
- All score fields are INTEGER
- All fantasy fields are REAL (use ROUND for display)

Common Patterns:
- Get player stats: JOIN player_game_stats pgs WITH players p ON pgs.player_id = p.player_id
- Get game info: JOIN games g ON pgs.game_id = g.game_id
- Career totals: Use SUM() and GROUP BY player_id
- Season stats: Filter WHERE g.season = X
- Win rate: SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END)
- Overtime games: WHERE g.overtime = 'Yes' (not = 1)
- Tied games: WHERE g.is_tie = 1 (this one IS boolean)
"""


def parse_sql_statements(sql_text):
    """
    Parse multiple SQL statements from text
    Returns list of individual SQL statements
    Validates each statement is complete
    """
    # Remove comments
    sql_text = re.sub(r'--.*$', '', sql_text, flags=re.MULTILINE)
    sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
    
    # Split by semicolons
    statements = []
    current = []
    in_string = False
    string_char = None
    paren_depth = 0
    
    for char in sql_text:
        if char in ('"', "'") and not in_string:
            in_string = True
            string_char = char
        elif char == string_char and in_string:
            in_string = False
            string_char = None
        elif char == '(' and not in_string:
            paren_depth += 1
        elif char == ')' and not in_string:
            paren_depth -= 1
        elif char == ';' and not in_string and paren_depth == 0:
            stmt = ''.join(current).strip()
            if stmt:
                # Validate statement
                if stmt.count('(') == stmt.count(')'):
                    statements.append(stmt)
                else:
                    # Incomplete statement - skip it
                    print(f"Skipping incomplete statement: {stmt[:100]}...")
            current = []
            continue
        
        current.append(char)
    
    # Add final statement
    stmt = ''.join(current).strip()
    if stmt and stmt.count('(') == stmt.count(')'):
        statements.append(stmt)
    
    return statements


def is_cte_query(sql):
    """Check if query uses Common Table Expressions"""
    return bool(re.search(r'\bWITH\b', sql, re.IGNORECASE))


# ============================================================================
# GOOGLE GEMINI
# ============================================================================

def generate_sql_gemini(question, api_key, model="gemini-2.5-flash"):
    """Generate SQL using Google Gemini API - supports multiple queries and CTEs"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        schema = get_schema_context()
        
        prompt = f"""You are an expert SQL analyst. Generate SQLite queries to answer complex questions about football statistics.

{schema}

User Question: {question}

CRITICAL OUTPUT FORMATTING RULES:
- Return ONLY pure SQL code - absolutely NO comments, NO headers, NO descriptions
- Use >= not ≥, use <= not ≤, use != not ≠
- NO markdown formatting (no **, no ##, no ```)
- NO text like "Query 1:", "**Description**:", etc.
- Start directly with SELECT or WITH
- Separate multiple queries with semicolon and newline only
- Each query must be complete and end with semicolon

QUERY DESIGN RULES:

**DEFAULT TO PER-GAME STATS** unless the question explicitly asks for:
- "career", "total", "all-time", "overall" → Use SUM() and GROUP BY player_id
- "season" or "in season X" → Use SUM() GROUP BY player_id with WHERE g.season = X
- Otherwise → Show individual game records with game context (player, game, date)

**FOR COMPLEX MULTI-PART QUESTIONS:**
- Break into 2-5 focused, INDEPENDENT queries
- Each query MUST stand alone (can run independently)
- CTEs (WITH clauses) ONLY work within a single query - they do NOT persist across queries
- If you need same data in multiple queries, repeat the CTE or subquery in each
- Prioritize the most important parts of the question

**ADVANCED FEATURES YOU CAN USE:**
- Common Table Expressions (WITH clause) - but only within ONE query
- Multiple queries separated by semicolons (each must be independent)
- Window functions (ROW_NUMBER, RANK, LAG, LEAD, etc.)
- Subqueries and complex joins
- CASE statements for conditional logic

**BEST PRACTICES:**
- Always use LIMIT 10-20 to show top results (even for "who has the most")
- Round decimals: ROUND(value, 1)
- Handle NULLs: NULLIF(sum, 0) or COALESCE(value, 0)
- Always JOIN with players table for player names: JOIN players p ON pgs.player_id = p.player_id
- Include game context for per-game stats: JOIN games g ON pgs.game_id = g.game_id
- Use meaningful column aliases: passing_yards AS PassYds

EXAMPLES OF CORRECT OUTPUT:

Example 1 - Per-Game Stats (DEFAULT for "who has the most"):
SELECT 
    p.player_name,
    'S' || g.season || '.G' || g.game_code as Game,
    g.game_date,
    pgs.receiving_yards,
    pgs.receiving_tds
FROM player_game_stats pgs
JOIN players p ON pgs.player_id = p.player_id
JOIN games g ON pgs.game_id = g.game_id
WHERE pgs.receiving_yards > 0
ORDER BY pgs.receiving_yards DESC
LIMIT 15;

Example 2 - Career Totals (when asked):
SELECT 
    p.player_name,
    COUNT(*) as Games,
    SUM(pgs.passing_yards) as CareerPassYds,
    SUM(pgs.passing_tds) as CareerPassTD,
    ROUND(SUM(pgs.completions) * 100.0 / NULLIF(SUM(pgs.attempts), 0), 1) as CompPct
FROM player_game_stats pgs
JOIN players p ON pgs.player_id = p.player_id
WHERE pgs.attempts > 0
GROUP BY p.player_id
ORDER BY CareerPassYds DESC
LIMIT 10;

Example 3 - Using CTE (all in ONE query):
WITH career_totals AS (
    SELECT 
        p.player_id,
        p.player_name,
        SUM(pgs.total_fantasy) as total
    FROM player_game_stats pgs
    JOIN players p ON pgs.player_id = p.player_id
    GROUP BY p.player_id
)
SELECT player_name, total 
FROM career_totals 
ORDER BY total DESC 
LIMIT 10;

Example 4 - Multiple INDEPENDENT Queries:
SELECT p.player_name, SUM(pgs.passing_yards) as Yards FROM player_game_stats pgs JOIN players p ON pgs.player_id = p.player_id GROUP BY p.player_id ORDER BY Yards DESC LIMIT 5;
SELECT p.player_name, SUM(pgs.receiving_yards) as Yards FROM player_game_stats pgs JOIN players p ON pgs.player_id = p.player_id GROUP BY p.player_id ORDER BY Yards DESC LIMIT 5;

Example 5 - Window Functions:
SELECT 
    p.player_name,
    pgs.total_fantasy,
    ROW_NUMBER() OVER (ORDER BY pgs.total_fantasy DESC) as rank
FROM player_game_stats pgs
JOIN players p ON pgs.player_id = p.player_id
ORDER BY rank
LIMIT 10;

Example 6 - Complex Analysis with CTE (all in one query):
WITH elite_games AS (
    SELECT 
        pgs.player_id,
        COUNT(*) as elite_count
    FROM player_game_stats pgs
    WHERE pgs.total_fantasy >= 40
    GROUP BY pgs.player_id
)
SELECT 
    p.player_name,
    eg.elite_count,
    COUNT(pgs.game_id) as total_games,
    ROUND(eg.elite_count * 100.0 / COUNT(pgs.game_id), 1) as elite_pct
FROM player_game_stats pgs
JOIN players p ON pgs.player_id = p.player_id
JOIN elite_games eg ON pgs.player_id = eg.player_id
GROUP BY p.player_id
ORDER BY elite_count DESC
LIMIT 10;

NOW GENERATE SQL FOR THE USER'S QUESTION.

Remember: 
- NO comments or headers
- Start with SELECT or WITH
- Pure SQL only
- Each query independent

SQL:"""
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8000,
                "stopSequences": []
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        
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
            
            if finish_reason == 'MAX_TOKENS':
                return None, "Response too long - query was cut off. Try simplifying your question."
            
            if finish_reason not in ['STOP', 'MAX_TOKENS']:
                return None, f"Response blocked. Reason: {finish_reason}. Try rephrasing."
            
            sql = candidate['content']['parts'][0]['text'].strip()
            
            # Aggressive cleanup
            sql = sql.replace('```sql', '').replace('```', '').replace('```SQL', '').strip()
            
            # Remove quotes
            if sql.startswith('"') and sql.endswith('"'):
                sql = sql[1:-1]
            if sql.startswith("'") and sql.endswith("'"):
                sql = sql[1:-1]
            
            # Remove markdown formatting
            sql = re.sub(r'\*\*.*?\*\*\n?', '', sql)
            sql = re.sub(r'##.*?$', '', sql, flags=re.MULTILINE)
            sql = re.sub(r'^Query \d+:.*?$', '', sql, flags=re.MULTILINE)
            sql = re.sub(r'^\*\*Query.*?$', '', sql, flags=re.MULTILINE)
            
            # Replace unicode operators
            sql = sql.replace('≥', '>=')
            sql = sql.replace('≤', '<=')
            sql = sql.replace('≠', '!=')
            sql = sql.replace('"', '"').replace('"', '"')
            sql = sql.replace(''', "'").replace(''', "'")
            
            # Remove leading garbage before first keyword
            first_keyword = re.search(r'\b(SELECT|WITH)\b', sql, re.IGNORECASE)
            if first_keyword:
                sql = sql[first_keyword.start():]
            
            # Remove non-SQL lines
            lines = sql.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                
                # Skip obviously non-SQL lines
                if stripped.startswith('**') or stripped.startswith('##'):
                    continue
                if stripped.startswith('Query ') and ':' in stripped:
                    continue
                if not stripped:
                    cleaned_lines.append('')
                    continue
                
                # Keep SQL lines
                cleaned_lines.append(line)
            
            sql = '\n'.join(cleaned_lines).strip()
            
            # Final validation
            if not re.search(r'\b(SELECT|WITH)\b', sql, re.IGNORECASE):
                return None, "Generated SQL doesn't contain SELECT or WITH. Response may be invalid."
            
            # Validate basic structure
            if sql.count('(') != sql.count(')'):
                return None, f"Generated SQL has unmatched parentheses ({sql.count('(')} open, {sql.count(')')} close). Try a simpler question."
            
            return sql, None
            
        except (KeyError, IndexError) as e:
            return None, f"Error parsing response: {e}"
        
    except Exception as e:
        return None, f"Error: {str(e)}"



def interpret_results_gemini(question, sql_queries, all_results, api_key, model="gemini-2.5-flash"):
    """
    Interpret multiple query results using Google Gemini API
    Fixed to handle None values in results
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
        
        # Build results text, handling None values
        results_text = ""
        successful_queries = 0
        
        for i, (sql, df) in enumerate(zip(sql_queries, all_results), 1):
            results_text += f"\n--- Query {i} ---\n"
            
            if df is not None and not df.empty:
                successful_queries += 1
                results_text += f"Results ({len(df)} rows):\n"
                results_text += df.head(15).to_string(index=False)
                results_text += "\n\n"
            elif df is not None and df.empty:
                results_text += "No results found\n\n"
            else:
                results_text += "Query failed\n\n"
        
        if successful_queries == 0:
            return None, "No successful queries to interpret"
        
        prompt = f"""You are analyzing football statistics query results.

Original Question: {question}

{successful_queries} of {len(sql_queries)} queries succeeded:

{results_text}

INSTRUCTIONS:
- Provide a comprehensive answer to the original question
- Use data from the SUCCESSFUL queries only
- Include specific numbers and player names
- Synthesize information across all result sets
- Be conversational and enthusiastic
- If some queries failed, work with what you have

Answer:"""
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 5000  # INCREASED from 2000
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
            return None, f"API Error {response.status_code}"
        
        result = response.json()
        
        try:
            interpretation = result['candidates'][0]['content']['parts'][0]['text'].strip()
            return interpretation, None
        except:
            return None, "Error parsing response"
        
    except Exception as e:
        return None, f"Error: {str(e)}"



def test_api_key(api_key, model="gemini-1.5-flash", provider="gemini"):
    """Test if the API key is valid"""
    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": "Say 'Working'"}]}]}
            response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "API key is valid!"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"
