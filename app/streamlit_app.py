"""
HHFL Stats Analytics Dashboard
Main Streamlit application
"""
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="HHFL Stats Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection
@st.cache_resource
def get_database_connection():
    """Create database connection"""
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
    .stat-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        text-transform: uppercase;
    }
    .ai-answer {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏈 HHFL Stats")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "👤 Player Lookup", "🏆 Leaderboards", "📊 Game Browser", 
     "📈 Trends", "⚔️ Head-to-Head", "🤖 AI Query"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")

# Quick stats in sidebar
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM games")
total_games = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM players")
total_players = cursor.fetchone()[0]

cursor.execute("SELECT MAX(season) FROM games WHERE season IS NOT NULL")
latest_season = cursor.fetchone()[0]

st.sidebar.metric("Total Games", f"{total_games:,}")
st.sidebar.metric("Total Players", f"{total_players:,}")
st.sidebar.metric("Latest Season", f"Season {latest_season}")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏈 HHFL Stats Dashboard</p>', unsafe_allow_html=True)
    
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
    
    st.subheader("📅 Recent Games")
    
    recent_games = pd.read_sql_query('''
        SELECT 
            season || '.' || game_code as Game,
            game_date as Date,
            captain1 || ' vs ' || captain2 as Matchup,
            CAST(team1_score as TEXT) || '-' || CAST(team2_score as TEXT) as Score,
            mvp as MVP,
            CASE WHEN overtime = 'Yes' THEN 'OT' ELSE '' END as OT
        FROM games
        WHERE season IS NOT NULL
        ORDER BY season DESC, game_number DESC
        LIMIT 10
    ''', conn)
    
    st.dataframe(recent_games, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏆 Season {latest_season} Top Fantasy Performers")
        
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
        st.subheader("🔥 All-Time Single Game Performances")
        
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
    
    st.subheader("🏅 Most MVP Awards")
    
    mvp_leaders = pd.read_sql_query('''
        SELECT mvp as Player, COUNT(*) as MVPs
        FROM games
        WHERE mvp IS NOT NULL AND mvp != ''
        GROUP BY mvp
        ORDER BY MVPs DESC
        LIMIT 10
    ''', conn)
    
    fig = px.bar(mvp_leaders, x='Player', y='MVPs', 
                 title='MVP Award Leaders',
                 color='MVPs',
                 color_continuous_scale='blues')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PLAYER LOOKUP PAGE
# ============================================================================
elif page == "👤 Player Lookup":
    st.markdown('<p class="main-header">👤 Player Lookup</p>', unsafe_allow_html=True)
    
    players_df = pd.read_sql_query('SELECT player_name FROM players ORDER BY player_name', conn)
    player_list = players_df['player_name'].tolist()
    
    selected_player = st.selectbox("Select a player:", player_list, 
                                   index=player_list.index('Jimmy Quinn') if 'Jimmy Quinn' in player_list else 0)
    
    if selected_player:
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
        
        st.subheader(f"📊 {selected_player} - Career Overview")
        
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
        
        st.subheader("📋 Game Log")
        
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
        
        st.dataframe(game_log, use_container_width=True, hide_index=True)
        
        st.subheader("📈 Fantasy Points Trend")
        
        trend_data = pd.read_sql_query(f'''
            SELECT 
                g.season,
                g.game_number,
                g.season || '.' || g.game_code as game_label,
                pgs.total_fantasy,
                pgs.offensive_fantasy,
                pgs.defensive_fantasy
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE p.player_name = ?
            ORDER BY g.season, g.game_number
        ''', conn, params=[selected_player])
        
        if len(trend_data) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(trend_data))),
                y=trend_data['total_fantasy'],
                mode='lines+markers',
                name='Total Fantasy',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{text}<br>Fantasy: %{y:.1f}<extra></extra>',
                text=trend_data['game_label']
            ))
            
            fig.update_layout(
                title=f"{selected_player} - Fantasy Points Over Career",
                xaxis_title="Game Number",
                yaxis_title="Fantasy Points",
                hovermode='closest',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# LEADERBOARDS PAGE
# ============================================================================
elif page == "🏆 Leaderboards":
    st.markdown('<p class="main-header">🏆 Leaderboards</p>', unsafe_allow_html=True)
    
    category = st.selectbox(
        "Select Category:",
        ["Fantasy Points", "Passing Yards", "Passing TDs", "Receiving Yards", "Receiving TDs", 
         "Rushing Yards", "Tackles", "Interceptions", "Sacks"]
    )
    
    time_period = st.radio("Time Period:", ["All-Time", "Single Season", "Single Game"], horizontal=True)
    
    season_filter = None
    if time_period == "Single Season":
        cursor.execute("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
        seasons = [row[0] for row in cursor.fetchall()]
        season_filter = st.selectbox("Select Season:", seasons)
    
    st.markdown("---")
    
    if category == "Fantasy Points":
        if time_period == "Single Game":
            query = '''
                SELECT 
                    p.player_name as Player,
                    'S' || g.season || '.G' || g.game_code as Game,
                    g.game_date as Date,
                    ROUND(pgs.total_fantasy, 1) as Fantasy,
                    ROUND(pgs.offensive_fantasy, 1) as "Off Pts",
                    ROUND(pgs.defensive_fantasy, 1) as "Def Pts"
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                ORDER BY pgs.total_fantasy DESC
                LIMIT 50
            '''
        elif time_period == "Single Season" and season_filter:
            query = f'''
                SELECT 
                    p.player_name as Player,
                    COUNT(*) as Games,
                    ROUND(SUM(pgs.total_fantasy), 1) as Total,
                    ROUND(AVG(pgs.total_fantasy), 1) as Avg,
                    ROUND(MAX(pgs.total_fantasy), 1) as Best
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE g.season = {season_filter}
                GROUP BY p.player_id
                ORDER BY Total DESC
                LIMIT 50
            '''
        else:
            query = '''
                SELECT 
                    p.player_name as Player,
                    COUNT(*) as Games,
                    ROUND(SUM(pgs.total_fantasy), 1) as Total,
                    ROUND(AVG(pgs.total_fantasy), 1) as Avg,
                    SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END) || '-' || 
                    SUM(CASE WHEN win_loss = 'Loss' THEN 1 ELSE 0 END) as Record
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                GROUP BY p.player_id
                HAVING Total > 0
                ORDER BY Total DESC
                LIMIT 50
            '''
    
    elif category == "Passing Yards":
        if time_period == "Single Game":
            query = '''
                SELECT 
                    p.player_name as Player,
                    'S' || g.season || '.G' || g.game_code as Game,
                    pgs.passing_yards as Yards,
                    pgs.completions || '/' || pgs.attempts as "Comp/Att",
                    pgs.passing_tds as TDs,
                    pgs.interceptions_thrown as INTs
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.passing_yards > 0
                ORDER BY pgs.passing_yards DESC
                LIMIT 50
            '''
        else:
            where_clause = f"WHERE g.season = {season_filter}" if season_filter else ""
            query = f'''
                SELECT 
                    p.player_name as Player,
                    SUM(pgs.passing_yards) as Yards,
                    SUM(pgs.completions) || '/' || SUM(pgs.attempts) as "Comp/Att",
                    ROUND(SUM(pgs.completions) * 100.0 / NULLIF(SUM(pgs.attempts), 0), 1) as "Comp%",
                    SUM(pgs.passing_tds) as TDs,
                    SUM(pgs.interceptions_thrown) as INTs
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                {where_clause}
                GROUP BY p.player_id
                HAVING Yards > 0
                ORDER BY Yards DESC
                LIMIT 50
            '''
    
    elif category == "Receiving Yards":
        if time_period == "Single Game":
            query = '''
                SELECT 
                    p.player_name as Player,
                    'S' || g.season || '.G' || g.game_code as Game,
                    pgs.receiving_yards as Yards,
                    pgs.receptions as Catches,
                    pgs.receiving_tds as TDs
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.receiving_yards > 0
                ORDER BY pgs.receiving_yards DESC
                LIMIT 50
            '''
        else:
            where_clause = f"WHERE g.season = {season_filter}" if season_filter else ""
            query = f'''
                SELECT 
                    p.player_name as Player,
                    SUM(pgs.receiving_yards) as Yards,
                    SUM(pgs.receptions) as Catches,
                    ROUND(SUM(pgs.receiving_yards) * 1.0 / NULLIF(SUM(pgs.receptions), 0), 1) as YPR,
                    SUM(pgs.receiving_tds) as TDs
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                {where_clause}
                GROUP BY p.player_id
                HAVING Yards > 0
                ORDER BY Yards DESC
                LIMIT 50
            '''
    
    elif category == "Tackles":
        if time_period == "Single Game":
            query = '''
                SELECT 
                    p.player_name as Player,
                    'S' || g.season || '.G' || g.game_code as Game,
                    pgs.tackles as Tackles,
                    pgs.sacks as Sacks,
                    pgs.interceptions as INTs
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                WHERE pgs.tackles > 0
                ORDER BY pgs.tackles DESC
                LIMIT 50
            '''
        else:
            where_clause = f"WHERE g.season = {season_filter}" if season_filter else ""
            query = f'''
                SELECT 
                    p.player_name as Player,
                    SUM(pgs.tackles) as Tackles,
                    SUM(pgs.sacks) as Sacks,
                    SUM(pgs.interceptions) as INTs,
                    ROUND(SUM(pgs.defensive_fantasy), 1) as "Def Pts"
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                JOIN games g ON pgs.game_id = g.game_id
                {where_clause}
                GROUP BY p.player_id
                HAVING Tackles > 0
                ORDER BY Tackles DESC
                LIMIT 50
            '''
    else:
        query = f'''SELECT 'Category: {category}' as Message'''
    
    df = pd.read_sql_query(query, conn)
    st.dataframe(df, use_container_width=True, hide_index=True, height=600)

# ============================================================================
# GAME BROWSER PAGE
# ============================================================================
elif page == "📊 Game Browser":
    st.markdown('<p class="main-header">📊 Game Browser</p>', unsafe_allow_html=True)
    
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

# ============================================================================
# TRENDS PAGE
# ============================================================================
elif page == "📈 Trends":
    st.markdown('<p class="main-header">📈 Trends & Analytics</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Average Score by Season")
    
    season_trends = pd.read_sql_query('''
        SELECT 
            season,
            COUNT(*) as games,
            ROUND(AVG(team1_score + team2_score), 1) as avg_total
        FROM games
        WHERE season IS NOT NULL AND season > 0
        GROUP BY season
        ORDER BY season
    ''', conn)
    
    fig = px.line(season_trends, x='season', y='avg_total',
                  title='Average Total Points Per Game by Season',
                  markers=True,
                  labels={'season': 'Season', 'avg_total': 'Avg Points'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 League Growth - Games Per Season")
    
    fig = px.bar(season_trends, x='season', y='games',
                 title='Number of Games Played Per Season',
                 labels={'season': 'Season', 'games': 'Games Played'},
                 color='games',
                 color_continuous_scale='blues')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# HEAD-TO-HEAD PAGE
# ============================================================================
elif page == "⚔️ Head-to-Head":
    st.markdown('<p class="main-header">⚔️ Head-to-Head Records</p>', unsafe_allow_html=True)
    
    cursor.execute('''
        SELECT DISTINCT captain1 FROM games WHERE captain1 IS NOT NULL 
        UNION 
        SELECT DISTINCT captain2 FROM games WHERE captain2 IS NOT NULL 
        ORDER BY 1
    ''')
    captains = [row[0] for row in cursor.fetchall()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        captain1 = st.selectbox("Captain 1:", captains, 
                               index=captains.index('Jimmy Quinn') if 'Jimmy Quinn' in captains else 0)
    
    with col2:
        captain2 = st.selectbox("Captain 2:", captains, 
                               index=captains.index('Troy Fite') if 'Troy Fite' in captains else 1)
    
    if captain1 and captain2 and captain1 != captain2:
        h2h_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_games,
                SUM(CASE WHEN (captain1 = ? AND team1_score > team2_score) OR 
                             (captain2 = ? AND team2_score > team1_score) THEN 1 ELSE 0 END) as c1_wins,
                SUM(CASE WHEN (captain1 = ? AND team1_score < team2_score) OR 
                             (captain2 = ? AND team2_score < team1_score) THEN 1 ELSE 0 END) as c2_wins,
                SUM(CASE WHEN team1_score = team2_score THEN 1 ELSE 0 END) as ties
            FROM games
            WHERE (captain1 = ? AND captain2 = ?) OR (captain1 = ? AND captain2 = ?)
        ''', conn, params=[captain1, captain1, captain2, captain2, captain1, captain2, captain2, captain1])
        
        stats = h2h_stats.iloc[0]
        
        if stats['total_games'] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{captain1} Wins", int(stats['c1_wins']))
            
            with col2:
                st.metric("Total Games", int(stats['total_games']))
                st.metric("Ties", int(stats['ties']))
            
            with col3:
                st.metric(f"{captain2} Wins", int(stats['c2_wins']))
            
            win_data = pd.DataFrame({
                'Captain': [captain1, captain2],
                'Wins': [stats['c1_wins'], stats['c2_wins']]
            })
            
            fig = px.pie(win_data, values='Wins', names='Captain',
                        title=f'{captain1} vs {captain2}')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# AI QUERY PAGE
# ============================================================================
elif page == "🤖 AI Query":
    st.markdown('<p class="main-header">AI-Powered Query</p>', unsafe_allow_html=True)
    
    st.markdown("Ask questions in plain English! AI will generate SQL, run it, and explain the results.")
    
    # Import LLM functions
    try:
        from app.llm_integration import (
            generate_sql_gemini,
            interpret_results_gemini,
            generate_sql_openai,
            interpret_results_openai,
            get_schema_context,
            test_api_key
        )
    except ImportError as e:
        st.error(f"LLM module error: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Provider selection
    provider = st.selectbox(
        "AI Provider:",
        ["Google Gemini (Free - Rate Limited)", "OpenAI (Paid)", "Manual (No API)"]
    )
    
    use_api = provider != "Manual (No API)"
    
    if provider == "Google Gemini (Free - Rate Limited)":
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("Gemini API Key:", type="password")
        with col2:
            model = st.selectbox("Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])
        
        st.caption("⚠️ Free tier: 15/min, 1500/day. You hit the limit - try OpenAI or Manual mode.")
        
    elif provider == "OpenAI (Paid)":
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("OpenAI API Key:", type="password")
        with col2:
            model = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
        
        st.caption("$5 free credits for new accounts (about 10,000 queries with gpt-3.5-turbo)")
    
    else:
        api_key = None
        model = None
        st.info("Manual mode: Copy/paste to ChatGPT/Claude/Gemini web (all free)")
    
    st.markdown("---")
    
    # Quick templates
    st.subheader("Quick Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Top Fantasy"):
            st.session_state.auto_question = "Who are the top 15 players by total career fantasy points?"
            st.rerun()
        if st.button("Best QBs"):
            st.session_state.auto_question = "Show me the top 10 quarterbacks by passing yards"
            st.rerun()
    
    with col2:
        if st.button("Top Receivers"):
            st.session_state.auto_question = "Who are the top 10 receivers by career yards?"
            st.rerun()
        if st.button("Best Defense"):
            st.session_state.auto_question = "Show the top 10 players by tackles"
            st.rerun()
    
    with col3:
        if st.button("MVP Leaders"):
            st.session_state.auto_question = "Who has won the most MVP awards?"
            st.rerun()
        if st.button("Best Games"):
            st.session_state.auto_question = "What are the top 10 single game fantasy performances?"
            st.rerun()
    
    st.markdown("---")
    
    # Initialize session state for question if needed
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ''
    
    # Check for auto-question from templates
    if 'auto_question' in st.session_state:
        st.session_state.current_question = st.session_state.auto_question
        del st.session_state.auto_question
    
    # Question input
    question = st.text_area(
        "Ask your question:",
        placeholder="e.g., Who has the most career receiving touchdowns?",
        height=100,
        value=st.session_state.current_question
    )
    
    # Update session state when question changes
    if question != st.session_state.current_question:
        st.session_state.current_question = question
    
    # Manual SQL input (always visible)
    manual_sql = st.text_area(
        "Or paste SQL directly:",
        placeholder="Paste your SQL query here...",
        height=150
    )
    
    # Run button
    if st.button("Run Query", type="primary", use_container_width=True):
        # Use the session state question, not the widget value
        current_question = st.session_state.current_question
        
        if not current_question and not manual_sql:
            st.warning("Please enter a question or paste SQL!")
        elif use_api and not api_key and not manual_sql:
            st.warning("Please enter your API key or use Manual mode!")
        else:
            sql_query = None
            
            # Priority: Use manual SQL if provided
            if manual_sql and manual_sql.strip():
                sql_query = manual_sql.strip()
                st.info("Using your manual SQL")
            
            # Otherwise generate SQL
            elif provider == "Google Gemini (Free - Rate Limited)" and api_key:
                with st.spinner("Generating SQL..."):
                    sql_query, error = generate_sql_gemini(current_question, api_key, model)
                    if error:
                        st.error(f"Error: {error}")
                        st.stop()
                    st.success("SQL generated!")
            
            elif provider == "OpenAI (Paid)" and api_key:
                with st.spinner("Generating SQL..."):
                    sql_query, error = generate_sql_openai(current_question, api_key, model)
                    if error:
                        st.error(f"Error: {error}")
                        st.stop()
                    st.success("SQL generated!")
            
            elif provider == "Manual (No API)":
                # Show prompt to help generate SQL
                if current_question:
                    schema = get_schema_context()
                    sql_prompt = f"""Generate SQLite query: {current_question}

{schema}

IMPORTANT: Use LIMIT 10 or more to show top results (even for "who has the most" questions).

Return ONLY the SQL query."""
                    
                    st.subheader("Copy this to ChatGPT/Claude/Gemini:")
                    st.code(sql_prompt, language="text")
                    st.info("After getting the SQL, paste it in the 'Or paste SQL directly' box above and click Run Query again")
                    st.stop()
                else:
                    st.warning("Please enter a question or paste SQL directly!")
                    st.stop()
            
            # Execute SQL
            if sql_query and sql_query.strip():
                st.markdown("---")
                
                with st.expander("View SQL Query", expanded=False):
                    st.code(sql_query, language="sql")
                
                try:
                    with st.spinner("Running query..."):
                        results_df = pd.read_sql_query(sql_query, conn)
                    
                    st.success(f"Found {len(results_df)} result(s)")
                    
                    st.subheader("Results:")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Interpret results
                    if not results_df.empty:
                        st.markdown("---")
                        
                        # Auto interpretation if using API
                        if provider == "Google Gemini (Free - Rate Limited)" and api_key:
                            with st.spinner("Interpreting..."):
                                interpretation, error = interpret_results_gemini(current_question, sql_query, results_df, api_key, model)
                                if interpretation:
                                    st.subheader("AI Answer:")
                                    st.markdown(f'<div class="ai-answer">{interpretation}</div>', unsafe_allow_html=True)
                                elif error:
                                    st.warning(f"Interpretation error: {error}")
                        
                        elif provider == "OpenAI (Paid)" and api_key:
                            with st.spinner("Interpreting..."):
                                interpretation, error = interpret_results_openai(current_question, sql_query, results_df, api_key, model)
                                if interpretation:
                                    st.subheader("AI Answer:")
                                    st.markdown(f'<div class="ai-answer">{interpretation}</div>', unsafe_allow_html=True)
                                elif error:
                                    st.warning(f"Interpretation error: {error}")
                        
                        else:
                            # Manual interpretation option
                            results_text = results_df.head(20).to_string(index=False)
                            
                            interp_prompt = f"""Question: {current_question}

Results (top results shown):
{results_text}

Answer the specific question ("who has the MOST"), but provide context from the other top results too.

Answer:"""
                            
                            with st.expander("Get AI interpretation"):
                                st.subheader("Copy this to AI:")
                                st.code(interp_prompt, language="text")
                                
                                interpretation = st.text_area("Paste answer:", height=150, key="manual_interp")
                                
                                if interpretation:
                                    st.markdown(f'<div class="ai-answer">{interpretation}</div>', unsafe_allow_html=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download CSV", data=csv, 
                                     file_name="results.csv", mime="text/csv")
                    
                except Exception as e:
                    st.error(f"SQL Error: {e}")
                    st.code(str(e))


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 About")
st.sidebar.info(f"""
HHFL Stats Dashboard

{total_games} games | {total_players} players

Built with Streamlit
""")