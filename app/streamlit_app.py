"""
HHFL Stats Analytics Dashboard
Streamlined design with tabs, no sidebar, no icons
"""
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="HHFL Stats Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Database connection
@st.cache_resource
def get_database_connection():
    return sqlite3.connect('data/processed/hhfl_stats.db', check_same_thread=False)

conn = get_database_connection()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ai-answer {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    button[data-baseweb="tab"] {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        font-size: 1.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for secrets
def get_api_key_from_secrets(provider="gemini"):
    """Get API key from Streamlit secrets"""
    try:
        if provider == "gemini":
            return st.secrets.get("GEMINI_API_KEY", None)
        elif provider == "openai":
            return st.secrets.get("OPENAI_API_KEY", None)
    except:
        return None

# Get basic stats
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM games")
total_games = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM players")
total_players = cursor.fetchone()[0]

cursor.execute("SELECT MAX(season) FROM games WHERE season IS NOT NULL")
latest_season = cursor.fetchone()[0]

# Header with logo
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    try:
        st.image("assets/logo.png", width=1000)
    except:
        # Fallback if image not found
        st.markdown('<p class="main-header">HHFL Stats Dashboard</p>', unsafe_allow_html=True)

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Player Lookup", "AI Query", "Records", "Stat Importer"])

# ============================================================================
# HOME TAB
# ============================================================================
with tab1:
    st.header("Home")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{total_games:,}")
    
    with col2:
        st.metric("Total Players", f"{total_players:,}")
    
    with col3:
        cursor.execute("SELECT COUNT(*) FROM player_game_stats")
        st.metric("Player Records", f"{cursor.fetchone()[0]:,}")
    
    with col4:
        cursor.execute("SELECT SUM(team1_score + team2_score) FROM games")
        st.metric("Total Points Scored", f"{cursor.fetchone()[0]:,}")
    
    st.markdown("---")
    
    # Game Browser (moved from separate page)
    st.subheader("Game Browser")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cursor.execute("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
        seasons = [row[0] for row in cursor.fetchall()]
        selected_season = st.selectbox("Season:", ["All"] + seasons)
    
    with col2:
        if selected_season != "All":
            cursor.execute(f"SELECT DISTINCT game_code FROM games WHERE season = {selected_season} ORDER BY game_number")
            games = [row[0] for row in cursor.fetchall()]
            selected_game = st.selectbox("Game:", ["All"] + games)
        else:
            selected_game = "All"
    
    with col3:
        cursor.execute("SELECT DISTINCT captain1 FROM games WHERE captain1 IS NOT NULL UNION SELECT DISTINCT captain2 FROM games WHERE captain2 IS NOT NULL ORDER BY 1")
        captains = [row[0] for row in cursor.fetchall()]
        selected_captain = st.selectbox("Captain:", ["All"] + captains)
    
    # Build query
    where_clauses = []
    if selected_season != "All":
        where_clauses.append(f"g.season = {selected_season}")
    if selected_game != "All" and selected_season != "All":
        where_clauses.append(f"g.game_code = '{selected_game}'")
    if selected_captain != "All":
        where_clauses.append(f"(g.captain1 = '{selected_captain}' OR g.captain2 = '{selected_captain}')")
    
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    games_query = f'''
        SELECT 
            g.season || '.' || g.game_code as Game,
            g.game_date as Date,
            g.captain1 as "Captain 1",
            g.captain2 as "Captain 2",
            g.team1_score || '-' || g.team2_score as Score,
            g.mvp as MVP,
            CASE WHEN g.overtime = 'Yes' THEN 'Yes' ELSE '' END as OT
        FROM games g
        {where_sql}
        ORDER BY g.season DESC, g.game_number DESC
    '''
    
    games_df = pd.read_sql_query(games_query, conn)
    st.dataframe(games_df, use_container_width=True, hide_index=True, height=500)
    
    # Top performers this season
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Season {latest_season} Top Fantasy Performers")
        
        top_season = pd.read_sql_query(f'''
            SELECT 
                p.player_name as Player,
                COUNT(*) as Games,
                ROUND(SUM(pgs.total_fantasy), 1) as Total,
                ROUND(AVG(pgs.total_fantasy), 1) as Avg
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE g.season = {latest_season}
            GROUP BY p.player_id
            ORDER BY Total DESC
            LIMIT 10
        ''', conn)
        
        st.dataframe(top_season, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("All-Time Single Game Performances")
        
        top_games = pd.read_sql_query('''
            SELECT 
                p.player_name as Player,
                'S' || g.season || '.G' || g.game_code as Game,
                ROUND(pgs.total_fantasy, 1) as Fantasy
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            ORDER BY pgs.total_fantasy DESC
            LIMIT 10
        ''', conn)
        
        st.dataframe(top_games, use_container_width=True, hide_index=True)

# ============================================================================
# PLAYER LOOKUP TAB
# ============================================================================
with tab2:
    st.header("Player Lookup")
    
    # Get all players
    players_df = pd.read_sql_query('SELECT player_name FROM players ORDER BY player_name', conn)
    player_list = players_df['player_name'].tolist()
    
    # Player selection
    selected_player = st.selectbox("Select a player:", player_list, 
                                   index=player_list.index('Jimmy Quinn') if 'Jimmy Quinn' in player_list else 0)
    
    if selected_player:
        # Get player stats
        player_stats = pd.read_sql_query(f'''
            SELECT 
                COUNT(*) as games,
                SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN win_loss = 'Loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN win_loss = 'Tie' THEN 1 ELSE 0 END) as ties,
                SUM(completions) as comps,
                SUM(attempts) as atts,
                SUM(passing_yards) as pass_yds,
                SUM(passing_tds) as pass_tds,
                SUM(interceptions_thrown) as ints,
                SUM(rush_attempts) as rush_att,
                SUM(rush_yards) as rush_yds,
                SUM(rush_tds) as rush_tds,
                SUM(receptions) as recs,
                SUM(receiving_yards) as rec_yds,
                SUM(receiving_tds) as rec_tds,
                SUM(tackles) as tackles,
                SUM(sacks) as sacks,
                SUM(interceptions) as def_ints,
                SUM(int_tds) as int_tds,
                ROUND(SUM(total_fantasy), 1) as total_fantasy,
                ROUND(AVG(total_fantasy), 1) as avg_fantasy
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            WHERE p.player_name = ?
        ''', conn, params=[selected_player])
        
        stats = player_stats.iloc[0]
        
        # Display career overview
        st.subheader(f"{selected_player} - Career Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Games", f"{int(stats['games']):,}")
            win_pct = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0
            st.metric("Record", f"{int(stats['wins'])}-{int(stats['losses'])}-{int(stats['ties'])}")
            st.caption(f"{win_pct:.1f}% Win Rate")
        
        with col2:
            st.metric("Total Fantasy", f"{stats['total_fantasy']:,.1f}")
            st.metric("Avg Per Game", f"{stats['avg_fantasy']:.1f}")
        
        with col3:
            if stats['atts'] > 0:
                comp_pct = (stats['comps'] / stats['atts'] * 100)
                st.metric("Passing Yards", f"{int(stats['pass_yds']):,}")
                st.caption(f"{int(stats['comps'])}/{int(stats['atts'])} ({comp_pct:.1f}%)")
                st.metric("Pass TDs", f"{int(stats['pass_tds'])}")
                st.caption(f"{int(stats['ints'])} INTs")
            else:
                st.caption("No passing stats")
        
        with col4:
            if stats['recs'] > 0:
                ypr = stats['rec_yds'] / stats['recs'] if stats['recs'] > 0 else 0
                st.metric("Receiving Yards", f"{int(stats['rec_yds']):,}")
                st.caption(f"{int(stats['recs'])} catches ({ypr:.1f} ypr)")
                st.metric("Rec TDs", f"{int(stats['rec_tds'])}")
            else:
                st.caption("No receiving stats")
        
        with col5:
            if stats['tackles'] > 0 or stats['def_ints'] > 0:
                st.metric("Tackles", f"{int(stats['tackles'])}")
                st.metric("INTs", f"{int(stats['def_ints'])}")
                st.caption(f"{stats['sacks']:.0f} sacks")
            else:
                st.caption("No defensive stats")
        
        st.markdown("---")
        
        # Game log
        st.subheader("Game Log")
        
        game_log = pd.read_sql_query(f'''
            SELECT 
                g.season || '.' || g.game_code as Game,
                g.game_date as Date,
                g.captain1 || ' vs ' || g.captain2 as Matchup,
                pgs.win_loss as Result,
                ROUND(pgs.total_fantasy, 1) as Fantasy,
                pgs.passing_yards as PassYds,
                pgs.passing_tds as PassTD,
                pgs.receiving_yards as RecYds,
                pgs.receiving_tds as RecTD,
                pgs.tackles as Tack,
                pgs.interceptions as INT
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE p.player_name = ?
            ORDER BY g.season DESC, g.game_number DESC
        ''', conn, params=[selected_player])
        
        st.dataframe(game_log, use_container_width=True, hide_index=True, height=400)

# ============================================================================
# AI QUERY TAB
# ============================================================================
with tab3:
    st.header("AI Query")
    
    # Import LLM functions
    try:
        from app.llm_integration import (
            generate_sql_gemini,
            interpret_results_gemini,
            get_schema_context,
            parse_sql_statements
        )
    except ImportError as e:
        st.error(f"LLM module error: {e}")
        st.stop()
    
    # Get API key from secrets
    api_key = get_api_key_from_secrets("gemini")
    model = "gemini-2.5-flash"
    
    st.markdown("---")
    
    # Quick templates
    st.subheader("Quick Templates")
    
    template_questions = {
        "Top Fantasy": "Who are the top 15 players by total career fantasy points? Show their games played, win-loss record, and average per game.",
        "Best QBs": "Show me the top 10 quarterbacks by career passing yards with their completion percentage, TDs, and interceptions",
        "Top Receivers": "Who are the top 10 receivers by career yards? Include their catches, TDs, and yards per reception",
        "Best Defense": "Show the top 10 defensive players by career tackles with their sacks, interceptions, and defensive fantasy points",
        "Best Team": "I want to see which set of four players (including a quarterback, leading receivers, and top defensive players) that had the best win loss records (when on the same team) also include some of the top stats (per game) from the players when playing together. These sets of four may be just a subset of the overall team. Include a requirement for each player having at least 30 games played total and each set having at least 2 games together.",
        "MVP Stats": "Look at the game stats for the game MVPs versus the stats for the other players in that game and tell me which stats are most impactful for the MVP selection. Then evaluate our Fantasy point system and tell me what changes you would make to better align it with the MVP selection."
    }
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    selected_template = None
    
    with col1:
        if st.button("Top Fantasy", use_container_width=True, key="tmpl_fantasy"):
            selected_template = template_questions["Top Fantasy"]
    
    with col2:
        if st.button("Best QBs", use_container_width=True, key="tmpl_qbs"):
            selected_template = template_questions["Best QBs"]
    
    with col3:
        if st.button("Top Receivers", use_container_width=True, key="tmpl_rec"):
            selected_template = template_questions["Top Receivers"]
    
    with col4:
        if st.button("Best Defense", use_container_width=True, key="tmpl_def"):
            selected_template = template_questions["Best Defense"]
    
    with col5:
        if st.button("Best Team", use_container_width=True, key="tmpl_team"):
            selected_template = template_questions["Best Team"]
    
    with col6:
        if st.button("MVP Stats", use_container_width=True, key="tmpl_mvp_stats"):
            selected_template = template_questions["MVP Stats"]
    
    st.markdown("---")
    
    # Initialize question in session state
    if 'ai_question' not in st.session_state:
        st.session_state.ai_question = ''
    
    # Use template if selected
    question_value = st.session_state.ai_question
    if selected_template:
        question_value = selected_template
    
    # Question input
    question = st.text_area(
        "Ask your question:",
        placeholder="e.g., Who has the most career receiving touchdowns?",
        height=100,
        value=question_value,
        key="question_text_area"
    )
    
    # Update session state
    if question != st.session_state.ai_question:
        st.session_state.ai_question = question
    
    # Manual SQL override
    with st.expander("Advanced: Manual SQL Override"):
        manual_sql = st.text_area(
            "Paste SQL directly:",
            placeholder="Optional: Paste your own SQL query here...",
            height=150
        )
    
    # Run button
    if st.button("Run Query", type="primary", use_container_width=True):
        current_question = st.session_state.ai_question
        
        if not current_question and not manual_sql:
            st.warning("Please enter a question!")
            st.stop()
        
        sql_text = None
        
        # Get SQL
        if manual_sql and manual_sql.strip():
            sql_text = manual_sql.strip()
            st.info("Using your manual SQL")
        
        elif api_key:
            with st.spinner("Generating SQL..."):
                sql_text, error = generate_sql_gemini(current_question, api_key, model)
                if error:
                    st.error(f"Error: {error}")
                    st.stop()
                st.success("SQL generated!")
        
        else:
            st.error("AI queries not configured.")
            st.stop()
        
        # Parse and execute
        if sql_text:
            sql_statements = parse_sql_statements(sql_text)
            
            st.markdown("---")
            
            with st.expander(f"View SQL ({len(sql_statements)} statement{'s' if len(sql_statements) > 1 else ''})", expanded=False):
                for i, sql in enumerate(sql_statements, 1):
                    if len(sql_statements) > 1:
                        st.write(f"Query {i}:")
                    st.code(sql, language="sql")
                    if i < len(sql_statements):
                        st.markdown("---")
            
            # Execute all queries
            all_results = []
            all_errors = []
            
            for i, sql_query in enumerate(sql_statements, 1):
                try:
                    with st.spinner(f"Running query {i} of {len(sql_statements)}..."):
                        results_df = pd.read_sql_query(sql_query, conn)
                        all_results.append(results_df)
                        all_errors.append(None)
                except Exception as e:
                    all_results.append(None)
                    all_errors.append(str(e))
            
            # Check success
            success_count = sum(1 for r in all_results if r is not None)
            
            if success_count == 0:
                st.error("All queries failed!")
                for i, (sql_query, error) in enumerate(zip(sql_statements, all_errors), 1):
                    with st.expander(f"Query {i} Error", expanded=True):
                        st.error(f"Error: {error}")
                        st.code(sql_query, language="sql")
                st.stop()
            
            st.success(f"Executed {success_count} of {len(sql_statements)} successfully!")
            
            # Display results - COLLAPSED
            with st.expander("View Query Results", expanded=False):
                for i, (results_df, error) in enumerate(zip(all_results, all_errors), 1):
                    if len(sql_statements) > 1:
                        st.markdown(f"### Query {i} Results:")
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif results_df is not None:
                        if len(results_df) == 0:
                            st.info("No results found")
                        else:
                            st.write(f"Found {len(results_df)} rows")
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    if i < len(sql_statements):
                        st.markdown("---")
            
            # Interpret results
            successful_results = [r for r in all_results if r is not None]
            
            if successful_results and api_key:
                st.markdown("---")
                
                with st.spinner("Interpreting results..."):
                    valid_queries = [sql_statements[i] for i, r in enumerate(all_results) if r is not None]
                    valid_results = [r for r in all_results if r is not None]
                    
                    interpretation, error = interpret_results_gemini(
                        current_question, valid_queries, valid_results, api_key, model
                    )
                    
                    if interpretation:
                        st.subheader("AI Answer:")
                        st.markdown(f'<div class="ai-answer">{interpretation}</div>', 
                                  unsafe_allow_html=True)
                    elif error:
                        st.warning(f"Could not generate interpretation: {error}")
            
            # Download
            if len(successful_results) == 1:
                csv = successful_results[0].to_csv(index=False)
                st.download_button("Download Results", data=csv, file_name="results.csv", mime="text/csv")
            elif len(successful_results) > 1:
                combined_csv = ""
                for i, df in enumerate(successful_results, 1):
                    combined_csv += f"Query {i}\n"
                    combined_csv += df.to_csv(index=False)
                    combined_csv += "\n\n"
                st.download_button(f"Download All Results ({len(successful_results)} queries)", 
                                 data=combined_csv, file_name="results.csv", mime="text/csv")

# ============================================================================
# RECORDS TAB
# ============================================================================
with tab4:
    st.header("League Records")
    
    st.markdown("All-time and single-season records, automatically updated from the database.")
    
    # ========================================================================
    # PASSING RECORDS
    # ========================================================================
    
    st.subheader("Passing Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Single Game")
        
        # Most attempts
        attempts_sg = pd.read_sql_query('''
            SELECT p.player_name as Player, 
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.attempts as Attempts
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.attempts > 0
            ORDER BY pgs.attempts DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Attempts", 
                 f"{attempts_sg.iloc[0]['Player']} - {int(attempts_sg.iloc[0]['Attempts'])}",
                 f"{attempts_sg.iloc[0]['Game']}")
        
        # Most completions
        comps_sg = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.completions as Completions
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.completions > 0
            ORDER BY pgs.completions DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Completions",
                 f"{comps_sg.iloc[0]['Player']} - {int(comps_sg.iloc[0]['Completions'])}",
                 f"{comps_sg.iloc[0]['Game']}")
        
        # Most yards
        yards_sg = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.passing_yards as Yards
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.passing_yards > 0
            ORDER BY pgs.passing_yards DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Yards",
                 f"{yards_sg.iloc[0]['Player']} - {int(yards_sg.iloc[0]['Yards'])}",
                 f"{yards_sg.iloc[0]['Game']}")
        
        # Most TDs
        tds_sg = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.passing_tds as TDs
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.passing_tds > 0
            ORDER BY pgs.passing_tds DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most TD Passes",
                 f"{tds_sg.iloc[0]['Player']} - {int(tds_sg.iloc[0]['TDs'])}",
                 f"{tds_sg.iloc[0]['Game']}")
        
        # Most INTs
        ints_sg = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.interceptions_thrown as INTs
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.interceptions_thrown > 0
            ORDER BY pgs.interceptions_thrown DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Interceptions",
                 f"{ints_sg.iloc[0]['Player']} - {int(ints_sg.iloc[0]['INTs'])}",
                 f"{ints_sg.iloc[0]['Game']}")
    
    with col2:
        st.markdown("### Season")
        
        # Most attempts
        attempts_s = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   g.season as Season,
                   SUM(pgs.attempts) as Attempts
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY Attempts DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Attempts",
                 f"{attempts_s.iloc[0]['Player']} - {int(attempts_s.iloc[0]['Attempts'])}",
                 f"Season {int(attempts_s.iloc[0]['Season'])}")
        
        # Most completions
        comps_s = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   g.season as Season,
                   SUM(pgs.completions) as Completions
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY Completions DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Completions",
                 f"{comps_s.iloc[0]['Player']} - {int(comps_s.iloc[0]['Completions'])}",
                 f"Season {int(comps_s.iloc[0]['Season'])}")
        
        # Most yards
        yards_s = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   g.season as Season,
                   SUM(pgs.passing_yards) as Yards
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY Yards DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Yards",
                 f"{yards_s.iloc[0]['Player']} - {int(yards_s.iloc[0]['Yards'])}",
                 f"Season {int(yards_s.iloc[0]['Season'])}")
        
        # Most TDs
        tds_s = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   g.season as Season,
                   SUM(pgs.passing_tds) as TDs
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY TDs DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most TD Passes",
                 f"{tds_s.iloc[0]['Player']} - {int(tds_s.iloc[0]['TDs'])}",
                 f"Season {int(tds_s.iloc[0]['Season'])}")
        
        # Most INTs
        ints_s = pd.read_sql_query('''
            SELECT p.player_name as Player,
                   g.season as Season,
                   SUM(pgs.interceptions_thrown) as INTs
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY INTs DESC
            LIMIT 1
        ''', conn)
        
        st.metric("Most Interceptions",
                 f"{ints_s.iloc[0]['Player']} - {int(ints_s.iloc[0]['INTs'])}",
                 f"Season {int(ints_s.iloc[0]['Season'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # RECEIVING RECORDS
    # ========================================================================
    
    st.subheader("Receiving Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Single Game")
        
        recs_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.receptions
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.receptions > 0
            ORDER BY pgs.receptions DESC LIMIT 1
        ''', conn)
        st.metric("Most Receptions", f"{recs_sg.iloc[0]['player_name']} - {int(recs_sg.iloc[0]['receptions'])}", f"{recs_sg.iloc[0]['Game']}")
        
        rec_yds_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.receiving_yards
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.receiving_yards > 0
            ORDER BY pgs.receiving_yards DESC LIMIT 1
        ''', conn)
        st.metric("Most Yards", f"{rec_yds_sg.iloc[0]['player_name']} - {int(rec_yds_sg.iloc[0]['receiving_yards'])}", f"{rec_yds_sg.iloc[0]['Game']}")
        
        rec_tds_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.receiving_tds
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.receiving_tds > 0
            ORDER BY pgs.receiving_tds DESC LIMIT 1
        ''', conn)
        st.metric("Most TDs", f"{rec_tds_sg.iloc[0]['player_name']} - {int(rec_tds_sg.iloc[0]['receiving_tds'])}", f"{rec_tds_sg.iloc[0]['Game']}")
    
    with col2:
        st.markdown("### Season")
        
        recs_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.receptions) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Receptions", f"{recs_s.iloc[0]['player_name']} - {int(recs_s.iloc[0]['total'])}", f"Season {int(recs_s.iloc[0]['season'])}")
        
        rec_yds_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.receiving_yards) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Yards", f"{rec_yds_s.iloc[0]['player_name']} - {int(rec_yds_s.iloc[0]['total'])}", f"Season {int(rec_yds_s.iloc[0]['season'])}")
        
        rec_tds_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.receiving_tds) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most TDs", f"{rec_tds_s.iloc[0]['player_name']} - {int(rec_tds_s.iloc[0]['total'])}", f"Season {int(rec_tds_s.iloc[0]['season'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # RUSHING RECORDS
    # ========================================================================
    
    st.subheader("Rushing Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Single Game")
        
        rush_att_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.rush_attempts
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.rush_attempts > 0
            ORDER BY pgs.rush_attempts DESC LIMIT 1
        ''', conn)
        st.metric("Most Attempts", f"{rush_att_sg.iloc[0]['player_name']} - {int(rush_att_sg.iloc[0]['rush_attempts'])}", f"{rush_att_sg.iloc[0]['Game']}")
        
        rush_yds_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.rush_yards
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.rush_yards > 0
            ORDER BY pgs.rush_yards DESC LIMIT 1
        ''', conn)
        st.metric("Most Yards", f"{rush_yds_sg.iloc[0]['player_name']} - {int(rush_yds_sg.iloc[0]['rush_yards'])}", f"{rush_yds_sg.iloc[0]['Game']}")
        
        rush_tds_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.rush_tds
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.rush_tds > 0
            ORDER BY pgs.rush_tds DESC LIMIT 1
        ''', conn)
        st.metric("Most TDs", f"{rush_tds_sg.iloc[0]['player_name']} - {int(rush_tds_sg.iloc[0]['rush_tds'])}", f"{rush_tds_sg.iloc[0]['Game']}")
    
    with col2:
        st.markdown("### Season")
        
        rush_att_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.rush_attempts) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Attempts", f"{rush_att_s.iloc[0]['player_name']} - {int(rush_att_s.iloc[0]['total'])}", f"Season {int(rush_att_s.iloc[0]['season'])}")
        
        rush_yds_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.rush_yards) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Yards", f"{rush_yds_s.iloc[0]['player_name']} - {int(rush_yds_s.iloc[0]['total'])}", f"Season {int(rush_yds_s.iloc[0]['season'])}")
        
        rush_tds_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.rush_tds) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most TDs", f"{rush_tds_s.iloc[0]['player_name']} - {int(rush_tds_s.iloc[0]['total'])}", f"Season {int(rush_tds_s.iloc[0]['season'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # DEFENSIVE RECORDS
    # ========================================================================
    
    st.subheader("Defensive Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Single Game")
        
        tacks_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.tackles
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.tackles > 0
            ORDER BY pgs.tackles DESC LIMIT 1
        ''', conn)
        st.metric("Most Tackles", f"{tacks_sg.iloc[0]['player_name']} - {int(tacks_sg.iloc[0]['tackles'])}", f"{tacks_sg.iloc[0]['Game']}")
        
        sacks_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.sacks
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.sacks > 0
            ORDER BY pgs.sacks DESC LIMIT 1
        ''', conn)
        st.metric("Most Sacks", f"{sacks_sg.iloc[0]['player_name']} - {sacks_sg.iloc[0]['sacks']:.0f}", f"{sacks_sg.iloc[0]['Game']}")
        
        def_ints_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, pgs.interceptions
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.interceptions > 0
            ORDER BY pgs.interceptions DESC LIMIT 1
        ''', conn)
        st.metric("Most Interceptions", f"{def_ints_sg.iloc[0]['player_name']} - {int(def_ints_sg.iloc[0]['interceptions'])}", f"{def_ints_sg.iloc[0]['Game']}")
    
    with col2:
        st.markdown("### Season")
        
        tacks_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.tackles) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Tackles", f"{tacks_s.iloc[0]['player_name']} - {int(tacks_s.iloc[0]['total'])}", f"Season {int(tacks_s.iloc[0]['season'])}")
        
        sacks_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.sacks) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Sacks", f"{sacks_s.iloc[0]['player_name']} - {sacks_s.iloc[0]['total']:.0f}", f"Season {int(sacks_s.iloc[0]['season'])}")
        
        def_ints_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(pgs.interceptions) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Interceptions", f"{def_ints_s.iloc[0]['player_name']} - {int(def_ints_s.iloc[0]['total'])}", f"Season {int(def_ints_s.iloc[0]['season'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # FANTASY & SCORING RECORDS
    # ========================================================================
    
    st.subheader("Fantasy & Scoring Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Single Game")
        
        fantasy_sg = pd.read_sql_query('''
            SELECT p.player_name, 'S' || g.season || '.G' || g.game_code as Game, 
                   ROUND(pgs.total_fantasy, 1) as fantasy
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            ORDER BY pgs.total_fantasy DESC LIMIT 1
        ''', conn)
        st.metric("Highest Fantasy Score", f"{fantasy_sg.iloc[0]['player_name']} - {fantasy_sg.iloc[0]['fantasy']}", f"{fantasy_sg.iloc[0]['Game']}")
        
        high_score_sg = pd.read_sql_query('''
            SELECT 'S' || season || '.G' || game_code as Game, 
                   game_date,
                   (team1_score + team2_score) as total,
                   team1_score || '-' || team2_score as score
            FROM games
            ORDER BY (team1_score + team2_score) DESC LIMIT 1
        ''', conn)
        st.metric("Highest Scoring Game", f"{high_score_sg.iloc[0]['score']} ({int(high_score_sg.iloc[0]['total'])} pts)", f"{high_score_sg.iloc[0]['Game']}")
        
        blowout = pd.read_sql_query('''
            SELECT 'S' || season || '.G' || game_code as Game,
                   ABS(team1_score - team2_score) as diff,
                   team1_score || '-' || team2_score as score
            FROM games
            WHERE team1_score != team2_score
            ORDER BY diff DESC LIMIT 1
        ''', conn)
        st.metric("Biggest Blowout", f"{blowout.iloc[0]['score']} ({int(blowout.iloc[0]['diff'])} pt diff)", f"{blowout.iloc[0]['Game']}")
    
    with col2:
        st.markdown("### Season")
        
        fantasy_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, ROUND(SUM(pgs.total_fantasy), 1) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Fantasy Points", f"{fantasy_s.iloc[0]['player_name']} - {fantasy_s.iloc[0]['total']}", f"Season {int(fantasy_s.iloc[0]['season'])}")
        
        mvps_s = pd.read_sql_query('''
            SELECT mvp as player, season, COUNT(*) as count
            FROM games
            WHERE mvp IS NOT NULL AND mvp != ''
            GROUP BY mvp, season
            ORDER BY count DESC LIMIT 1
        ''', conn)
        st.metric("Most MVPs in Season", f"{mvps_s.iloc[0]['player']} - {int(mvps_s.iloc[0]['count'])}", f"Season {int(mvps_s.iloc[0]['season'])}")
        
        wins_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(CASE WHEN pgs.win_loss = 'Win' THEN 1 ELSE 0 END) as wins
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY wins DESC LIMIT 1
        ''', conn)
        st.metric("Most Wins", f"{wins_s.iloc[0]['player_name']} - {int(wins_s.iloc[0]['wins'])}", f"Season {int(wins_s.iloc[0]['season'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # CAREER RECORDS
    # ========================================================================
    
    st.subheader("Career Records")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most MVPs career
        mvps_career = pd.read_sql_query('''
            SELECT mvp as Player, COUNT(*) as MVPs
            FROM games
            WHERE mvp IS NOT NULL AND mvp != ''
            GROUP BY mvp
            ORDER BY MVPs DESC LIMIT 1
        ''', conn)
        st.metric("Most Career MVPs", f"{mvps_career.iloc[0]['Player']} - {int(mvps_career.iloc[0]['MVPs'])}")
        
        # Most games
        most_games = pd.read_sql_query('''
            SELECT p.player_name, COUNT(*) as games
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            GROUP BY p.player_id
            ORDER BY games DESC LIMIT 1
        ''', conn)
        st.metric("Most Games Played", f"{most_games.iloc[0]['player_name']} - {int(most_games.iloc[0]['games'])}")
    
    with col2:
        # Best win percentage (min 50 games)
        win_pct = pd.read_sql_query('''
            SELECT p.player_name,
                   SUM(CASE WHEN pgs.win_loss = 'Win' THEN 1 ELSE 0 END) as wins,
                   COUNT(*) as games,
                   ROUND(SUM(CASE WHEN pgs.win_loss = 'Win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            GROUP BY p.player_id
            HAVING COUNT(*) >= 50
            ORDER BY pct DESC LIMIT 1
        ''', conn)
        st.metric("Best Win % (50+ games)", f"{win_pct.iloc[0]['player_name']} - {win_pct.iloc[0]['pct']}%", 
                 f"{int(win_pct.iloc[0]['wins'])}-{int(win_pct.iloc[0]['games'] - win_pct.iloc[0]['wins'])}")
        
        # Most career fantasy
        career_fant = pd.read_sql_query('''
            SELECT p.player_name, ROUND(SUM(pgs.total_fantasy), 1) as total
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            GROUP BY p.player_id
            ORDER BY total DESC LIMIT 1
        ''', conn)
        st.metric("Most Career Fantasy", f"{career_fant.iloc[0]['player_name']} - {career_fant.iloc[0]['total']}")
    
    with col3:
        # Highest avg fantasy (min 50 games)
        avg_fant = pd.read_sql_query('''
            SELECT p.player_name, ROUND(AVG(pgs.total_fantasy), 1) as avg, COUNT(*) as games
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            GROUP BY p.player_id
            HAVING COUNT(*) >= 50
            ORDER BY avg DESC LIMIT 1
        ''', conn)
        st.metric("Highest Avg Fantasy (50+ games)", f"{avg_fant.iloc[0]['player_name']} - {avg_fant.iloc[0]['avg']}", 
                 f"{int(avg_fant.iloc[0]['games'])} games")
        
        # Most 40+ point games
        elite_games = pd.read_sql_query('''
            SELECT p.player_name, COUNT(*) as elite_games
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            WHERE pgs.total_fantasy >= 40
            GROUP BY p.player_id
            ORDER BY elite_games DESC LIMIT 1
        ''', conn)
        st.metric("Most 40+ Point Games", f"{elite_games.iloc[0]['player_name']} - {int(elite_games.iloc[0]['elite_games'])}")
    
    st.markdown("---")
    
    # ========================================================================
    # MISC RECORDS
    # ========================================================================
    
    st.subheader("Miscellaneous Records")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most OT games in a season
        ot_games = pd.read_sql_query('''
            SELECT season, COUNT(*) as ot_count
            FROM games
            WHERE overtime = 'Yes'
            GROUP BY season
            ORDER BY ot_count DESC LIMIT 1
        ''', conn)
        st.metric("Most OT Games in Season", f"{int(ot_games.iloc[0]['ot_count'])} games", f"Season {int(ot_games.iloc[0]['season'])}")
        
        # Best single-game completion percentage (min 20 attempts)
        comp_pct = pd.read_sql_query('''
            SELECT p.player_name, 
                   'S' || g.season || '.G' || g.game_code as Game,
                   pgs.completions,
                   pgs.attempts,
                   ROUND(pgs.completions * 100.0 / pgs.attempts, 1) as pct
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.attempts >= 20
            ORDER BY pct DESC LIMIT 1
        ''', conn)
        st.metric("Best Completion % (20+ att)", 
                 f"{comp_pct.iloc[0]['player_name']} - {comp_pct.iloc[0]['pct']:.1f}%",
                 f"{int(comp_pct.iloc[0]['completions'])}/{int(comp_pct.iloc[0]['attempts'])} in {comp_pct.iloc[0]['Game']}")
        
        # Most total TDs in a game (passing + receiving + rushing)
        total_tds = pd.read_sql_query('''
            SELECT p.player_name,
                   'S' || g.season || '.G' || g.game_code as Game,
                   (pgs.passing_tds + pgs.receiving_tds + pgs.rush_tds) as total_tds,
                   pgs.passing_tds as pass_td,
                   pgs.receiving_tds as rec_td,
                   pgs.rush_tds as rush_td
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE (pgs.passing_tds + pgs.receiving_tds + pgs.rush_tds) > 0
            ORDER BY total_tds DESC LIMIT 1
        ''', conn)
        st.metric("Most Total TDs in Game", 
                 f"{total_tds.iloc[0]['player_name']} - {int(total_tds.iloc[0]['total_tds'])} TDs",
                 f"{int(total_tds.iloc[0]['pass_td'])} pass, {int(total_tds.iloc[0]['rec_td'])} rec, {int(total_tds.iloc[0]['rush_td'])} rush - {total_tds.iloc[0]['Game']}")
    
    with col2:
        # Most consecutive wins
        consecutive_wins = pd.read_sql_query('''
            WITH numbered_games AS (
                SELECT 
                    p.player_name,
                    pgs.win_loss,
                    g.season,
                    g.game_number,
                    ROW_NUMBER() OVER (PARTITION BY p.player_id ORDER BY g.season, g.game_number) as game_num,
                    ROW_NUMBER() OVER (PARTITION BY p.player_id, pgs.win_loss ORDER BY g.season, g.game_number) as streak_num
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
            ),
            streaks AS (
                SELECT 
                    player_name,
                    win_loss,
                    COUNT(*) as streak_length,
                    MIN(season) as start_season,
                    MAX(season) as end_season
                FROM numbered_games
                WHERE win_loss = 'Win'
                GROUP BY player_name, (game_num - streak_num)
            )
            SELECT player_name, streak_length, start_season, end_season
            FROM streaks
            ORDER BY streak_length DESC LIMIT 1
        ''', conn)
        
        if len(consecutive_wins) > 0:
            st.metric("Most Consecutive Wins", 
                     f"{consecutive_wins.iloc[0]['player_name']} - {int(consecutive_wins.iloc[0]['streak_length'])} games",
                     f"S{int(consecutive_wins.iloc[0]['start_season'])}-S{int(consecutive_wins.iloc[0]['end_season'])}")
        else:
            st.caption("Calculating streaks...")
        
        # Most consecutive losses
        consecutive_losses = pd.read_sql_query('''
            WITH numbered_games AS (
                SELECT 
                    p.player_name,
                    pgs.win_loss,
                    g.season,
                    g.game_number,
                    ROW_NUMBER() OVER (PARTITION BY p.player_id ORDER BY g.season, g.game_number) as game_num,
                    ROW_NUMBER() OVER (PARTITION BY p.player_id, pgs.win_loss ORDER BY g.season, g.game_number) as streak_num
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
            ),
            streaks AS (
                SELECT 
                    player_name,
                    win_loss,
                    COUNT(*) as streak_length,
                    MIN(season) as start_season,
                    MAX(season) as end_season
                FROM numbered_games
                WHERE win_loss = 'Loss'
                GROUP BY player_name, (game_num - streak_num)
            )
            SELECT player_name, streak_length, start_season, end_season
            FROM streaks
            ORDER BY streak_length DESC LIMIT 1
        ''', conn)
        
        if len(consecutive_losses) > 0:
            st.metric("Most Consecutive Losses", 
                     f"{consecutive_losses.iloc[0]['player_name']} - {int(consecutive_losses.iloc[0]['streak_length'])} games",
                     f"S{int(consecutive_losses.iloc[0]['start_season'])}-S{int(consecutive_losses.iloc[0]['end_season'])}")
        else:
            st.caption("Calculating streaks...")
    
    with col3:
        # Best captain win percentage (min 20 games)
        captain_pct = pd.read_sql_query('''
            WITH captain_records AS (
                SELECT 
                    captain1 as captain,
                    COUNT(*) as games,
                    SUM(CASE WHEN team1_score > team2_score THEN 1 ELSE 0 END) as wins
                FROM games
                WHERE captain1 IS NOT NULL
                GROUP BY captain1
                
                UNION ALL
                
                SELECT 
                    captain2 as captain,
                    COUNT(*) as games,
                    SUM(CASE WHEN team2_score > team1_score THEN 1 ELSE 0 END) as wins
                FROM games
                WHERE captain2 IS NOT NULL
                GROUP BY captain2
            )
            SELECT 
                captain as Captain,
                SUM(games) as total_games,
                SUM(wins) as total_wins,
                ROUND(SUM(wins) * 100.0 / SUM(games), 1) as win_pct
            FROM captain_records
            GROUP BY captain
            HAVING SUM(games) >= 20
            ORDER BY win_pct DESC LIMIT 1
        ''', conn)
        
        st.metric("Best Captain Win % (20+ games)", 
                 f"{captain_pct.iloc[0]['Captain']} - {captain_pct.iloc[0]['win_pct']:.1f}%",
                 f"{int(captain_pct.iloc[0]['total_wins'])}-{int(captain_pct.iloc[0]['total_games'] - captain_pct.iloc[0]['total_wins'])}")
        
        # Most losses in a season
        losses_s = pd.read_sql_query('''
            SELECT p.player_name, g.season, SUM(CASE WHEN pgs.win_loss = 'Loss' THEN 1 ELSE 0 END) as losses
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            GROUP BY p.player_id, g.season
            ORDER BY losses DESC LIMIT 1
        ''', conn)
        st.metric("Most Losses in Season", 
                 f"{losses_s.iloc[0]['player_name']} - {int(losses_s.iloc[0]['losses'])}",
                 f"Season {int(losses_s.iloc[0]['season'])}")

# ============================================================================
# STAT IMPORTER TAB
# ============================================================================
with tab5:
    st.header("Stat Importer")
    
    st.markdown("Enter a new game using your familiar shorthand codes!")
    
    # Import play parser
    try:
        from app.play_parser import PlayParser
    except ImportError:
        st.error("Play parser module not found. Make sure play_parser.py exists in app/ folder.")
        st.stop()
    
    # Initialize session state
    if 'game_setup_complete' not in st.session_state:
        st.session_state.game_setup_complete = False
    if 'team1_roster' not in st.session_state:
        st.session_state.team1_roster = []
    if 'team2_roster' not in st.session_state:
        st.session_state.team2_roster = []
    if 'current_plays' not in st.session_state:
        st.session_state.current_plays = []
    if 'current_drive' not in st.session_state:
        st.session_state.current_drive = 1
    if 'current_down' not in st.session_state:
        st.session_state.current_down = 1
    if 'current_offense' not in st.session_state:
        st.session_state.current_offense = 1
    
    # Get all players
    cursor.execute("SELECT DISTINCT player_name FROM players ORDER BY player_name")
    all_players = [row[0] for row in cursor.fetchall()]
    
    # Game Setup
    with st.expander("Game Setup", expanded=not st.session_state.game_setup_complete):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            game_date = st.date_input("Game Date", value=datetime.now())
            season = st.number_input("Season", min_value=1, value=latest_season, step=1)
        
        with col2:
            game_number = st.number_input("Game Number", min_value=1, value=1, step=1)
            game_suffix = st.text_input("Game Suffix (optional)", max_chars=1, placeholder="A, B, C...")
        
        with col3:
            captain1 = st.selectbox("Captain 1:", all_players, 
                                   index=all_players.index('Jimmy Quinn') if 'Jimmy Quinn' in all_players else 0)
            captain2 = st.selectbox("Captain 2:", all_players,
                                   index=all_players.index('Troy Fite') if 'Troy Fite' in all_players else 1)
        
        mvp = st.selectbox("MVP (select after game):", [""] + all_players)
        overtime = st.checkbox("Overtime?")
        
        if st.button("Confirm Game Setup"):
            st.session_state.game_setup_complete = True
            st.rerun()
    
    # Team Rosters
    roster_expanded = (not st.session_state.team1_roster and not st.session_state.team2_roster) or st.session_state.get('roster_editing', False)
    
    with st.expander("Team Rosters", expanded=roster_expanded):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Team 1 - {captain1}**")
            
            if st.button("Load Jimmy Quinn's Team (Example)", key="load_jq"):
                st.session_state.team1_roster = [
                    {'code': 'Q', 'player': 'Jimmy Quinn'},
                    {'code': 'B', 'player': 'Beau Bromley'},
                    {'code': 'G', 'player': 'Garett Hill'},
                    {'code': 'R', 'player': 'Cody Rodriguez'},
                    {'code': 'L', 'player': 'Brandon Least'},
                    {'code': 'P', 'player': 'Terrance Pugh'},
                    {'code': 'A', 'player': 'Alex Couch'}
                ]
                st.session_state.roster_editing = True
                st.rerun()
            
            add_col1, add_col2 = st.columns([1, 3])
            with add_col1:
                player1_code = st.text_input("Code:", max_chars=1, key="code_t1", placeholder="Q")
            with add_col2:
                new_player1 = st.selectbox("Player:", [""] + all_players, key="add_t1")
            
            if st.button("Add to Team 1", use_container_width=True, key="add_btn_t1") and new_player1 and player1_code:
                if not any(p['code'] == player1_code for p in st.session_state.team1_roster):
                    st.session_state.team1_roster.append({'code': player1_code, 'player': new_player1})
                    st.session_state.roster_editing = True
                    st.rerun()
                else:
                    st.warning("Code already used!")
            
            if st.session_state.team1_roster:
                st.markdown("**Roster:**")
                for i, p in enumerate(st.session_state.team1_roster):
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.code(p['code'])
                    with col_b:
                        st.write(p['player'])
                    with col_c:
                        if st.button("X", key=f"del_t1_{i}", help="Remove"):
                            st.session_state.team1_roster.pop(i)
                            st.session_state.roster_editing = True
                            st.rerun()
        
        with col2:
            st.write(f"**Team 2 - {captain2}**")
            
            if st.button("Load Tyler Smith's Team (Example)", key="load_ts"):
                st.session_state.team2_roster = [
                    {'code': 'T', 'player': 'Tyler Smith'},
                    {'code': 'C', 'player': 'Chris Cross'},
                    {'code': 'S', 'player': 'Shane Gibson'},
                    {'code': 'D', 'player': 'Caleb Deck'},
                    {'code': 'Z', 'player': 'Cale Rodriguez'},
                    {'code': 'W', 'player': 'Brian Ward'},
                    {'code': 'E', 'player': 'Tyler E'}
                ]
                st.session_state.roster_editing = True
                st.rerun()
            
            add_col1, add_col2 = st.columns([1, 3])
            with add_col1:
                player2_code = st.text_input("Code:", max_chars=1, key="code_t2", placeholder="T")
            with add_col2:
                new_player2 = st.selectbox("Player:", [""] + all_players, key="add_t2")
            
            if st.button("Add to Team 2", use_container_width=True, key="add_btn_t2") and new_player2 and player2_code:
                if not any(p['code'] == player2_code for p in st.session_state.team2_roster):
                    st.session_state.team2_roster.append({'code': player2_code, 'player': new_player2})
                    st.session_state.roster_editing = True
                    st.rerun()
                else:
                    st.warning("Code already used!")
            
            if st.session_state.team2_roster:
                st.markdown("**Roster:**")
                for i, p in enumerate(st.session_state.team2_roster):
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.code(p['code'])
                    with col_b:
                        st.write(p['player'])
                    with col_c:
                        if st.button("X", key=f"del_t2_{i}", help="Remove"):
                            st.session_state.team2_roster.pop(i)
                            st.session_state.roster_editing = True
                            st.rerun()
        
        st.markdown("---")
        if st.session_state.team1_roster and st.session_state.team2_roster:
            if st.button("Done - Rosters Complete", use_container_width=True, type="primary"):
                st.session_state.roster_editing = False
                st.rerun()
    
    # Build player codes
    player_codes = {}
    for p in st.session_state.team1_roster + st.session_state.team2_roster:
        player_codes[p['code']] = p['player']
    
    # Play Entry
    st.markdown("---")
    st.subheader("Play-by-Play Entry")
    
    # Code Reference
    with st.expander("Code Reference Guide"):
        st.markdown("""
        ### Play Code Format
        
        **Pass Play:** `[QB Code][Receiver Code][Yards][Tackler Code]`
        - Example: `TS17L` = Tyler Smith to Shane Gibson for 17 yards, tackled by Brandon Least
        
        **Rush Play:** `[Rusher Code][Yards][Tackler Code]`
        - Example: `Q13S` = Jimmy Quinn run for 13 yards, tackled by Shane Gibson
        
        **Incomplete:** `[QB Code]`
        - Example: `T` = Tyler Smith incomplete pass
        
        **Special Plays:**
        - `[Code][Code] K` = Kickoff
        - `[QB][Receiver] I` = Interception
        - `[QB][Tackler] S` = Sack
        - `P` = Punt
        
        ### Your Player Codes:
        """)
        
        if player_codes:
            code_cols = st.columns(4)
            codes_list = list(player_codes.items())
            for i, (code, name) in enumerate(codes_list):
                with code_cols[i % 4]:
                    st.code(f"{code} = {name}")
        else:
            st.info("Add players to team rosters to see codes")
    
    if not player_codes:
        st.warning("Please add players to team rosters first!")
        st.stop()
    
    # Initialize parser
    parser = PlayParser(player_codes)
    
    # Play entry form
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_stat = st.text_input("Stat Code:", placeholder="e.g., TS17L, Q, CL", key="new_stat_code")
    
    with col2:
        new_detail = st.text_input("Detail:", placeholder="K, I, S, P", max_chars=2, key="new_detail_code")
    
    with col3:
        st.write("")
        st.write("")
        if st.button("Add Play", type="primary", use_container_width=True):
            if new_stat:
                parsed = parser.parse_play(new_stat, new_detail)
                parsed['drive'] = st.session_state.current_drive
                parsed['down'] = st.session_state.current_down
                parsed['offense_team'] = st.session_state.current_offense
                
                st.session_state.current_plays.append(parsed)
                
                # Update game state
                if parsed['is_turnover'] or parsed['play_type'] == 'Punt':
                    st.session_state.current_drive += 1
                    st.session_state.current_down = 1
                    st.session_state.current_offense = 2 if st.session_state.current_offense == 1 else 1
                elif parsed['is_touchdown']:
                    st.session_state.current_drive += 1
                    st.session_state.current_down = 1
                    st.session_state.current_offense = 2 if st.session_state.current_offense == 1 else 1
                elif st.session_state.current_down >= 6:
                    st.session_state.current_drive += 1
                    st.session_state.current_down = 1
                    st.session_state.current_offense = 2 if st.session_state.current_offense == 1 else 1
                else:
                    st.session_state.current_down += 1
                
                st.rerun()
    
    # Current Game Status
    if st.session_state.current_plays:
        player_stats, team1_score, team2_score = parser.calculate_game_stats(st.session_state.current_plays)
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(f"{captain1} Score", team1_score)
        with col2:
            st.metric(f"{captain2} Score", team2_score)
        with col3:
            st.metric("Total Plays", len(st.session_state.current_plays))
        with col4:
            st.metric(f"Drive {st.session_state.current_drive}", f"{st.session_state.current_down}th down")
        
        st.markdown("---")
        st.subheader("Recent Plays")
        
        recent_plays = st.session_state.current_plays[-10:]
        plays_display = []
        for i, play in enumerate(recent_plays):
            play_num = len(st.session_state.current_plays) - len(recent_plays) + i + 1
            plays_display.append({
                '#': play_num,
                'Code': play['stat_code'],
                'Description': play['description'],
                'Yards': play['yards'] if play['yards'] > 0 else '',
                'Drive': play['drive'],
                'Down': play['down']
            })
        
        plays_df = pd.DataFrame(plays_display)
        st.dataframe(plays_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("Current Player Stats")
        
        stats_list = []
        for player, stats in player_stats.items():
            if any(stats.values()):
                off_fantasy, def_fantasy, total_fantasy = parser.calculate_fantasy_points(stats)
                stats_list.append({
                    'Player': player,
                    'PassYds': stats['passing_yards'],
                    'PassTD': stats['passing_tds'],
                    'RecYds': stats['receiving_yards'],
                    'RecTD': stats['receiving_tds'],
                    'RushYds': stats['rush_yards'],
                    'Tack': stats['tackles'],
                    'INT': stats['interceptions'],
                    'Fantasy': total_fantasy
                })
        
        if stats_list:
            stats_df = pd.DataFrame(stats_list)
            stats_df = stats_df.sort_values('Fantasy', ascending=False)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Undo Last Play"):
                if st.session_state.current_plays:
                    st.session_state.current_plays.pop()
                    st.rerun()
        
        with col2:
            if st.button("Start New Drive"):
                st.session_state.current_drive += 1
                st.session_state.current_down = 1
                st.rerun()
        
        with col3:
            if st.button("Clear All"):
                st.session_state.current_plays = []
                st.session_state.team1_roster = []
                st.session_state.team2_roster = []
                st.session_state.current_drive = 1
                st.session_state.current_down = 1
                st.session_state.game_setup_complete = False
                st.rerun()
        
        with col4:
            if st.button("Save Game", type="primary"):
                with st.spinner("Saving..."):
                    try:
                        cursor = conn.cursor()
                        
                        cursor.execute('''
                            INSERT INTO games 
                            (game_date, season, game_number, game_suffix, game_code,
                             captain1, captain2, mvp, team1_score, team2_score, overtime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_date.strftime('%Y-%m-%d'), season, game_number,
                            game_suffix if game_suffix else None,
                            f"{game_number}{game_suffix}" if game_suffix else str(game_number),
                            captain1, captain2, mvp if mvp else None,
                            team1_score, team2_score, 'Yes' if overtime else 'No'
                        ))
                        
                        game_id = cursor.lastrowid
                        
                        for player_name, stats in player_stats.items():
                            cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (player_name,))
                            player_result = cursor.fetchone()
                            if not player_result:
                                continue
                            player_id = player_result[0]
                            
                            off_fantasy, def_fantasy, total_fantasy = parser.calculate_fantasy_points(stats)
                            
                            on_team1 = any(p['player'] == player_name for p in st.session_state.team1_roster)
                            win_loss = 'Win' if (on_team1 and team1_score > team2_score) or (not on_team1 and team2_score > team1_score) else ('Loss' if (on_team1 and team1_score < team2_score) or (not on_team1 and team2_score < team1_score) else 'Tie')
                            
                            comp_ratio = stats['completions'] / stats['attempts'] if stats['attempts'] > 0 else 0
                            
                            cursor.execute('''
                                INSERT INTO player_game_stats (
                                    game_id, player_id, completions, attempts, completion_ratio,
                                    passing_yards, passing_tds, interceptions_thrown,
                                    rush_attempts, rush_yards, rush_tds,
                                    receptions, receiving_yards, receiving_tds,
                                    tackles, sacks, interceptions, int_tds,
                                    win_loss, offensive_fantasy, defensive_fantasy, total_fantasy
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                game_id, player_id, stats['completions'], stats['attempts'], comp_ratio,
                                stats['passing_yards'], stats['passing_tds'], stats['interceptions_thrown'],
                                stats['rush_attempts'], stats['rush_yards'], stats['rush_tds'],
                                stats['receptions'], stats['receiving_yards'], stats['receiving_tds'],
                                stats['tackles'], stats['sacks'], stats['interceptions'], stats['int_tds'],
                                win_loss, off_fantasy, def_fantasy, total_fantasy
                            ))
                        
                        for play_num, play in enumerate(st.session_state.current_plays, start=1):
                            qb_id = receiver_id = tackler_id = rusher_id = None
                            
                            if play['qb']:
                                cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (play['qb'],))
                                result = cursor.fetchone()
                                qb_id = result[0] if result else None
                            if play['receiver']:
                                cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (play['receiver'],))
                                result = cursor.fetchone()
                                receiver_id = result[0] if result else None
                            if play['tackler']:
                                cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (play['tackler'],))
                                result = cursor.fetchone()
                                tackler_id = result[0] if result else None
                            if play['rusher']:
                                cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (play['rusher'],))
                                result = cursor.fetchone()
                                rusher_id = result[0] if result else None
                            
                            cursor.execute('''
                                INSERT INTO plays (
                                    game_id, play_number, stat_code, detail_code, yards,
                                    play_description, drive_number, down_number, play_type, offense_team,
                                    is_touchdown, is_incomplete, is_fumble, is_interception, is_safety, is_turnover,
                                    qb_id, rusher_id, receiver_id, tackler_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                game_id, play_num, play['stat_code'], play['detail_code'], play['yards'],
                                play['description'], play['drive'], play['down'], play['play_type'], play['offense_team'],
                                play['is_touchdown'], play['is_incomplete'], play['is_fumble'], 
                                play['is_interception'], play['is_safety'], play['is_turnover'],
                                qb_id, rusher_id, receiver_id, tackler_id
                            ))
                        
                        conn.commit()
                        st.success(f"Game saved! Season {season}.{game_number}{game_suffix or ''}")
                        st.balloons()
                        
                        if st.button("Start New Game"):
                            st.session_state.current_plays = []
                            st.session_state.team1_roster = []
                            st.session_state.team2_roster = []
                            st.session_state.current_drive = 1
                            st.session_state.current_down = 1
                            st.session_state.game_setup_complete = False
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error saving game: {e}")
                        conn.rollback()
    
    else:
        st.info("Enter play codes above to start tracking the game!")


# Footer
st.markdown("---")
st.caption(f"HHFL Stats Dashboard | {total_games} games | {total_players} players | Built with Streamlit")
