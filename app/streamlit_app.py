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

def get_api_key_from_secrets(provider="gemini"):
    """Get API key from Streamlit secrets (server-side only)"""
    try:
        if provider == "gemini":
            return st.secrets.get("GEMINI_API_KEY", None)
        elif provider == "openai":
            return st.secrets.get("OPENAI_API_KEY", None)
    except FileNotFoundError:
        # Secrets file doesn't exist (local development without secrets)
        return None
    except Exception:
        return None

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
     "📈 Trends", "⚔️ Head-to-Head", "🤖 AI Query", "📝 Stat Importer"]  # Added here
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
    
    # Import LLM functions
    try:
        from app.llm_integration import (
            generate_sql_gemini,
            interpret_results_gemini,
            get_schema_context
        )
    except ImportError as e:
        st.error(f"LLM module error: {e}")
        st.stop()
    
    # Get API key from secrets (no user input needed)
    api_key = get_api_key_from_secrets("gemini")
    model = "gemini-2.5-flash"
    
    st.markdown("---")
    
    # Quick templates - ONE ROW
    st.subheader("Quick Templates")
    
    template_questions = {
        "Top Fantasy": "Who are the top 15 players by total career fantasy points?",
        "Best QBs": "Show me the top 10 quarterbacks by passing yards",
        "Top Receivers": "Who are the top 10 receivers by career yards?",
        "Best Defense": "Show the top 10 players by tackles",
        "MVP Leaders": "Who has won the most MVP awards?",
        "Best Games": "What are the top 10 single game fantasy performances?"
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
        if st.button("MVP Leaders", use_container_width=True, key="tmpl_mvp"):
            selected_template = template_questions["MVP Leaders"]
    
    with col6:
        if st.button("Best Games", use_container_width=True, key="tmpl_games"):
            selected_template = template_questions["Best Games"]
    
    st.markdown("---")
    
    # Question input - use selected template if clicked
    default_question = selected_template if selected_template else st.session_state.get('ai_question', '')
    
    question = st.text_area(
        "Ask your question:",
        placeholder="e.g., Who has the most career receiving touchdowns?",
        height=100,
        value=default_question
    )
    
    # Save to session state
    st.session_state.ai_question = question
    
    # Manual SQL override option (collapsible)
    with st.expander("Advanced: Manual SQL Override"):
        manual_sql = st.text_area(
            "Paste SQL directly:",
            placeholder="Optional: Paste your own SQL query here...",
            height=150
        )
    
    # Run button
    if st.button("Run Query", type="primary", use_container_width=True):
        # Use session state question
        current_question = st.session_state.ai_question
        
        if not current_question and not manual_sql:
            st.warning("Please enter a question!")
            st.stop()
        
        sql_query = None
        
        # Priority: Use manual SQL if provided
        if manual_sql and manual_sql.strip():
            sql_query = manual_sql.strip()
            st.info("Using your manual SQL")
        
        # Otherwise generate SQL with AI
        elif api_key:
            with st.spinner("Generating SQL..."):
                sql_query, error = generate_sql_gemini(current_question, api_key, model)
                if error:
                    st.error(f"Error: {error}")
                    st.stop()
                st.success("SQL generated!")
        
        else:
            st.error("AI queries not configured.")
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
                
                # Interpret results with AI
                if not results_df.empty and api_key:
                    st.markdown("---")
                    
                    with st.spinner("Interpreting results..."):
                        interpretation, error = interpret_results_gemini(
                            current_question, sql_query, results_df, api_key, model
                        )
                        
                        if interpretation:
                            st.subheader("AI Answer:")
                            st.markdown(f'<div class="ai-answer">{interpretation}</div>', 
                                      unsafe_allow_html=True)
                        elif error:
                            st.warning(f"Could not generate interpretation: {error}")
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV", 
                    data=csv, 
                    file_name="query_results.csv", 
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"SQL Error: {e}")
                st.code(str(e))

# ============================================================================
# STAT IMPORTER PAGE
# ============================================================================
elif page == "📝 Stat Importer":
    st.markdown('<p class="main-header">📝 Game Stat Importer</p>', unsafe_allow_html=True)
    
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
    
    # Get all players for dropdowns
    cursor.execute("SELECT DISTINCT player_name FROM players ORDER BY player_name")
    all_players = [row[0] for row in cursor.fetchall()]
    
    # ========================================================================
    # GAME SETUP SECTION
    # ========================================================================
    
    with st.expander("⚙️ Game Setup", expanded=not st.session_state.game_setup_complete):
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
        
        if st.button("✅ Confirm Game Setup"):
            st.session_state.game_setup_complete = True
            st.rerun()
    
    # ========================================================================
    # TEAM ROSTERS SECTION
    # ========================================================================
    
    # Keep roster expander open if rosters are being edited
    roster_expanded = (not st.session_state.team1_roster and not st.session_state.team2_roster) or st.session_state.get('roster_editing', False)
    
    with st.expander("👥 Team Rosters", expanded=roster_expanded):
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
                        if st.button("🗑️", key=f"del_t1_{i}", help="Remove player"):
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
                        if st.button("🗑️", key=f"del_t2_{i}", help="Remove player"):
                            st.session_state.team2_roster.pop(i)
                            st.session_state.roster_editing = True
                            st.rerun()
        
        # Done button to collapse
        st.markdown("---")
        if st.session_state.team1_roster and st.session_state.team2_roster:
            if st.button("✅ Done - Rosters Complete", use_container_width=True, type="primary"):
                st.session_state.roster_editing = False
                st.rerun()


    
    # ========================================================================
    # BUILD PLAYER CODES - MOVED BEFORE PLAY ENTRY
    # ========================================================================
    
    # Build player code lookup
    player_codes = {}
    for p in st.session_state.team1_roster + st.session_state.team2_roster:
        player_codes[p['code']] = p['player']
    
    # ========================================================================
    # PLAY ENTRY SECTION
    # ========================================================================
    
    st.markdown("---")
    st.subheader("⚡ Play-by-Play Entry")
    
    # CODE REFERENCE GUIDE
    with st.expander("📖 Code Reference Guide"):
        st.markdown("""
        ### Play Code Format
        
        **Pass Play:** `[QB Code][Receiver Code][Yards][Tackler Code]`
        - Example: `TS17L` = Tyler Smith to Shane Gibson for 17 yards, tackled by Brandon Least
        
        **Rush Play:** `[Rusher Code][Yards][Tackler Code]`
        - Example: `Q13S` = Jimmy Quinn run for 13 yards, tackled by Shane Gibson
        
        **Incomplete:** `[QB Code]`
        - Example: `T` = Tyler Smith incomplete pass
        
        **Special Plays:**
        - `[Code][Code] K` = Kickoff (tackler + returner)
        - `[QB][Receiver] I` = Interception
        - `[QB][Tackler] S` = Sack
        - `P` = Punt
        
        **Touchdowns:**
        - System auto-detects based on scoring
        
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
    
    # Check if rosters are set
    if not player_codes:
        st.warning("⚠️ Please add players to team rosters first!")
        st.stop()
    
    # Initialize parser
    parser = PlayParser(player_codes)
    
    # Play entry form
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_stat = st.text_input(
            "Stat Code:",
            placeholder="e.g., TS17L, Q, CL",
            key="new_stat_code",
            help="Your shorthand play code"
        )
    
    with col2:
        new_detail = st.text_input(
            "Detail (optional):",
            placeholder="K, I, S, P",
            max_chars=2,
            key="new_detail_code"
        )
    
    with col3:
        st.write("")
        st.write("")
        if st.button("➕ Add Play", type="primary", use_container_width=True):
            if new_stat:
                parsed = parser.parse_play(new_stat, new_detail)
                
                parsed['drive'] = st.session_state.current_drive
                parsed['down'] = st.session_state.current_down
                parsed['offense_team'] = st.session_state.current_offense
                
                st.session_state.current_plays.append(parsed)
                
                # Update drive/down logic
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
    
    # ========================================================================
    # CURRENT GAME STATUS
    # ========================================================================
    
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
        st.subheader("📋 Recent Plays")
        
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
        st.subheader("📊 Current Player Stats")
        
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
            if st.button("◀️ Undo Last Play"):
                if st.session_state.current_plays:
                    st.session_state.current_plays.pop()
                    st.rerun()
        
        with col2:
            if st.button("🔄 Start New Drive"):
                st.session_state.current_drive += 1
                st.session_state.current_down = 1
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All"):
                st.session_state.current_plays = []
                st.session_state.team1_roster = []
                st.session_state.team2_roster = []
                st.session_state.current_drive = 1
                st.session_state.current_down = 1
                st.session_state.game_setup_complete = False
                st.rerun()
        
        with col4:
            if st.button("💾 Save Game", type="primary"):
                with st.spinner("Saving game to database..."):
                    try:
                        cursor = conn.cursor()
                        
                        cursor.execute('''
                            INSERT INTO games 
                            (game_date, season, game_number, game_suffix, game_code,
                             captain1, captain2, mvp, team1_score, team2_score, overtime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_date.strftime('%Y-%m-%d'),
                            season,
                            game_number,
                            game_suffix if game_suffix else None,
                            f"{game_number}{game_suffix}" if game_suffix else str(game_number),
                            captain1,
                            captain2,
                            mvp if mvp else None,
                            team1_score,
                            team2_score,
                            'Yes' if overtime else 'No'
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
                            if on_team1:
                                win_loss = 'Win' if team1_score > team2_score else ('Loss' if team1_score < team2_score else 'Tie')
                            else:
                                win_loss = 'Win' if team2_score > team1_score else ('Loss' if team2_score < team1_score else 'Tie')
                            
                            comp_ratio = stats['completions'] / stats['attempts'] if stats['attempts'] > 0 else 0
                            
                            cursor.execute('''
                                INSERT INTO player_game_stats (
                                    game_id, player_id,
                                    completions, attempts, completion_ratio,
                                    passing_yards, passing_tds, interceptions_thrown,
                                    rush_attempts, rush_yards, rush_tds,
                                    receptions, receiving_yards, receiving_tds,
                                    tackles, sacks, interceptions, int_tds,
                                    win_loss, offensive_fantasy, defensive_fantasy, total_fantasy
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                game_id, player_id,
                                stats['completions'], stats['attempts'], comp_ratio,
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
                        
                        st.success(f"✅ Game saved! Season {season}.{game_number}{game_suffix or ''}")
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
        st.info("👆 Enter play codes above to start tracking the game!")

    
    

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 About")
st.sidebar.info(f"""
HHFL Stats Dashboard

{total_games} games | {total_players} players

Built with Streamlit
""")