"""
Data loading utilities for HHFL Stats Database
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import re


def safe_int(val):
    """Safely convert value to integer"""
    if pd.isna(val) or val == '' or val == '-':
        return 0
    try:
        if isinstance(val, str):
            val = val.replace(',', '')
        return int(float(val))
    except:
        return 0


def safe_float(val):
    """Safely convert value to float"""
    if pd.isna(val) or val == '' or val == '-':
        return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(',', '')
        return float(val)
    except:
        return 0.0


def safe_str(val):
    """Safely convert value to string"""
    if pd.isna(val) or val == '':
        return None
    return str(val).strip()


def parse_game_number(game_val):
    """
    Parse game number that may have letter suffix (e.g., '13A', '5B')
    Returns tuple: (game_number, game_suffix)
    Examples: '13A' -> (13, 'A'), '5' -> (5, None), '1B' -> (1, 'B')
    """
    if pd.isna(game_val) or game_val == '':
        return None, None
    
    game_str = str(game_val).strip()
    
    # Match pattern: number optionally followed by letter(s)
    match = re.match(r'^(\d+)([A-Za-z]*)$', game_str)
    
    if match:
        game_num = int(match.group(1))
        suffix = match.group(2) if match.group(2) else None
        return game_num, suffix
    
    return None, None


def parse_date(date_val):
    """Parse various date formats including Excel serial dates"""
    if pd.isna(date_val) or date_val == '':
        return None
    
    if isinstance(date_val, datetime):
        return date_val.strftime('%Y-%m-%d')
    
    if isinstance(date_val, pd.Timestamp):
        return date_val.strftime('%Y-%m-%d')
    
    date_str = str(date_val).strip()
    
    if not date_str:
        return None
    
    date_formats = [
        '%m/%d/%Y',
        '%Y-%m-%d',
        '%m/%d/%y',
        '%d/%m/%Y',
        '%Y/%m/%d'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except:
            continue
    
    try:
        excel_date = int(float(date_str))
        if 25000 < excel_date < 100000:
            base_date = datetime(1899, 12, 30)
            return (base_date + timedelta(days=excel_date)).strftime('%Y-%m-%d')
    except:
        pass
    
    return None


def create_schema(cursor):
    """Create all database tables and indexes"""
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name TEXT UNIQUE NOT NULL,
        code TEXT,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Updated games table to handle game suffixes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_date DATE,
        season INTEGER,
        game_number INTEGER,
        game_suffix TEXT,
        game_code TEXT,
        captain1 TEXT,
        captain2 TEXT,
        mvp TEXT,
        team1_score INTEGER,
        team2_score INTEGER,
        overtime TEXT,
        is_tie BOOLEAN DEFAULT 0,
        comment TEXT,
        week_number INTEGER,
        season_game TEXT,
        UNIQUE(season, game_number, game_suffix)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS game_rosters (
        roster_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        player_id INTEGER,
        team_number INTEGER,
        is_captain BOOLEAN DEFAULT 0,
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        UNIQUE(game_id, player_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS player_game_stats (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        player_id INTEGER,
        captain TEXT,
        
        completions INTEGER DEFAULT 0,
        attempts INTEGER DEFAULT 0,
        completion_ratio REAL DEFAULT 0,
        passing_yards INTEGER DEFAULT 0,
        passing_tds INTEGER DEFAULT 0,
        interceptions_thrown INTEGER DEFAULT 0,
        qbr REAL DEFAULT 0,
        qb_avg_drive REAL DEFAULT 0,
        
        rush_attempts INTEGER DEFAULT 0,
        rush_yards INTEGER DEFAULT 0,
        rush_tds INTEGER DEFAULT 0,
        
        receptions INTEGER DEFAULT 0,
        receiving_yards INTEGER DEFAULT 0,
        receiving_tds INTEGER DEFAULT 0,
        
        tackles INTEGER DEFAULT 0,
        sacks REAL DEFAULT 0,
        interceptions INTEGER DEFAULT 0,
        int_tds INTEGER DEFAULT 0,
        fumble_recoveries INTEGER DEFAULT 0,
        fumble_tds INTEGER DEFAULT 0,
        
        kickoff_tds INTEGER DEFAULT 0,
        safeties INTEGER DEFAULT 0,
        
        win_loss TEXT,
        
        offensive_fantasy REAL DEFAULT 0,
        defensive_fantasy REAL DEFAULT 0,
        total_fantasy REAL DEFAULT 0,
        
        player_game_label TEXT,
        streak INTEGER DEFAULT 0,
        game_count INTEGER DEFAULT 0,
        
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id),
        UNIQUE(game_id, player_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plays (
        play_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        play_number INTEGER,
        stat_code TEXT,
        detail_code TEXT,
        yards INTEGER,
        play_description TEXT,
        drive_number INTEGER,
        down_number INTEGER,
        play_type TEXT,
        offense_team INTEGER,
        
        is_touchdown BOOLEAN DEFAULT 0,
        is_incomplete BOOLEAN DEFAULT 0,
        is_fumble BOOLEAN DEFAULT 0,
        is_interception BOOLEAN DEFAULT 0,
        is_safety BOOLEAN DEFAULT 0,
        is_turnover BOOLEAN DEFAULT 0,
        
        qb_id INTEGER,
        rusher_id INTEGER,
        receiver_id INTEGER,
        tackler_id INTEGER,
        returner_id INTEGER,
        fumble_recovery_id INTEGER,
        
        team1_score INTEGER,
        team2_score INTEGER,
        
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        FOREIGN KEY (qb_id) REFERENCES players(player_id),
        FOREIGN KEY (rusher_id) REFERENCES players(player_id),
        FOREIGN KEY (receiver_id) REFERENCES players(player_id),
        FOREIGN KEY (tackler_id) REFERENCES players(player_id),
        FOREIGN KEY (returner_id) REFERENCES players(player_id),
        FOREIGN KEY (fumble_recovery_id) REFERENCES players(player_id)
    )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games(season, game_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_game_stats(player_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_plays_game ON plays(game_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_plays_player ON plays(qb_id)')
    
    print("✅ Database schema created")


def load_games_from_file(file_path, cursor):
    """Load games data from tab-delimited file - handles game suffixes like 13A, 5B"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split('\t')
    
    # Find column indexes
    try:
        date_idx = header.index('Game Date')
        season_idx = header.index('Season')
        game_idx = header.index('Game')
        cap1_idx = header.index('Captain 1')
        cap2_idx = header.index('Captain 2')
        mvp_idx = header.index('MVP')
        t1_score_idx = header.index('T1 Score')
        t2_score_idx = header.index('T2 Score')
        ot_idx = header.index('OT?')
        tie_idx = header.index('Tie?')
        comment_idx = header.index('Comment')
        week_idx = header.index('Week #')
    except ValueError as e:
        print(f"❌ Error finding column: {e}")
        return 0
    
    games_loaded = 0
    games_skipped = 0
    skip_reasons = {}
    
    for line_num, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        
        parts = line.strip().split('\t')
        
        # Handle rows with different column counts due to missing leading date
        if parts[0].strip() and not '/' in parts[0] and parts[0].strip().isdigit():
            parts = [''] + parts
        
        # Ensure we have enough columns
        while len(parts) < max(date_idx, season_idx, game_idx, cap1_idx, cap2_idx, 
                               mvp_idx, t1_score_idx, t2_score_idx, ot_idx, tie_idx, 
                               comment_idx, week_idx) + 1:
            parts.append('')
        
        # Extract values
        season = safe_int(parts[season_idx])
        game_number, game_suffix = parse_game_number(parts[game_idx])
        
        # Skip if no valid season or game number
        if not season or not game_number:
            games_skipped += 1
            reason = f"Season={parts[season_idx]}, Game={parts[game_idx]}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        
        # Create game code (e.g., "13A" or "5")
        game_code = parts[game_idx].strip()
        
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO games 
                (game_date, season, game_number, game_suffix, game_code, captain1, captain2, mvp,
                 team1_score, team2_score, overtime, is_tie, comment, week_number, season_game)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                parse_date(parts[date_idx]),
                season,
                game_number,
                game_suffix,
                game_code,
                safe_str(parts[cap1_idx]),
                safe_str(parts[cap2_idx]),
                safe_str(parts[mvp_idx]),
                safe_int(parts[t1_score_idx]),
                safe_int(parts[t2_score_idx]),
                safe_str(parts[ot_idx]),
                safe_str(parts[tie_idx]) == 'Yes',
                safe_str(parts[comment_idx]),
                safe_int(parts[week_idx]),
                f"{season}.{game_code}"
            ))
            
            if cursor.rowcount > 0:
                games_loaded += 1
                
        except sqlite3.IntegrityError:
            # Duplicate - already exists
            pass
        except Exception as e:
            games_skipped += 1
            if games_skipped <= 3:
                print(f"   ⚠️  Line {line_num}: Error - {e}")
    
    if skip_reasons:
        print(f"\n   Skipped games by reason (showing first 10):")
        for i, (reason, count) in enumerate(list(skip_reasons.items())[:10], 1):
            print(f"      {reason}: {count} times")
    
    return games_loaded


def load_player_stats_from_file(file_path, cursor):
    """Load player game stats from tab-delimited file - handles game suffixes"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split('\t')
    
    # Find column indexes
    try:
        s_idx = header.index('S')
        g_idx = header.index('G')
        captain_idx = header.index('Captain')
        player_idx = header.index('Player')
        
        comps_idx = header.index('Comps')
        atts_idx = header.index('Atts')
        ratio_idx = header.index('Ratio')
        yards_idx = header.index('Yards')
        tds_idx = header.index('TDs')
        ints_idx = header.index('Ints')
        qbr_idx = header.index('QBR')
        qb_avg_idx = header.index('QB Avg Drive')
        
        rush_att_idx = header.index('Rush Att')
        rush_yds_idx = header.index('Rush Yards')
        rush_tds_idx = header.index('RushTDs')
        
        recs_idx = header.index('Recs')
        rec_yds_idx = header.index('RecYards')
        rec_tds_idx = header.index('Rec Tds')
        
        tacks_idx = header.index('Tacks')
        sacks_idx = header.index('Sacks')
        ints2_idx = header.index('Ints2')
        int_td_idx = header.index('Int TD')
        fum_rec_idx = header.index('Fum Rec')
        fum_td_idx = header.index('Fum TDs')
        ko_td_idx = header.index('KO TD')
        safety_idx = header.index('Safeties')
        
        wl_idx = header.index('Win/Loss')
        off_fant_idx = header.index('Off Fantasy')
        def_fant_idx = header.index('Def Fantasy')
        fant_idx = header.index('Fantasy')
        pg_label_idx = header.index('Player - Game')
        streak_idx = header.index('Streak')
        gc_idx = header.index('Game Count')
        
    except ValueError as e:
        print(f"❌ Error finding column: {e}")
        print(f"Available columns: {header}")
        return 0, 0
    
    # First pass: collect all unique players
    all_players = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        
        if len(parts) > player_idx:
            player = safe_str(parts[player_idx])
            if player:
                all_players.add(player)
    
    # Insert all players
    players_added = 0
    for player_name in sorted(all_players):
        try:
            cursor.execute('INSERT OR IGNORE INTO players (player_name) VALUES (?)', (player_name,))
            if cursor.rowcount > 0:
                players_added += 1
        except:
            pass
    
    print(f"✅ Added {players_added} new players (total unique: {len(all_players)})")
    
    # Second pass: load stats
    stats_loaded = 0
    stats_skipped = 0
    skip_reasons = {}
    
    for line_num, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        
        parts = line.strip().split('\t')
        
        # Ensure enough columns
        while len(parts) < max(s_idx, g_idx, player_idx, fant_idx, gc_idx) + 1:
            parts.append('')
        
        season = safe_int(parts[s_idx])
        game_str = parts[g_idx].strip()
        game_num, game_suffix = parse_game_number(game_str)
        player = safe_str(parts[player_idx])
        
        if not season:
            stats_skipped += 1
            skip_reasons['no_season'] = skip_reasons.get('no_season', 0) + 1
            continue
            
        if not game_num:
            stats_skipped += 1
            skip_reasons['no_game'] = skip_reasons.get('no_game', 0) + 1
            continue
            
        if not player:
            stats_skipped += 1
            skip_reasons['no_player'] = skip_reasons.get('no_player', 0) + 1
            continue
        
        # Get game_id - now matching with suffix
        if game_suffix:
            cursor.execute('''
                SELECT game_id FROM games 
                WHERE season = ? AND game_number = ? AND game_suffix = ?
            ''', (season, game_num, game_suffix))
        else:
            cursor.execute('''
                SELECT game_id FROM games 
                WHERE season = ? AND game_number = ? AND (game_suffix IS NULL OR game_suffix = '')
            ''', (season, game_num))
        
        game_result = cursor.fetchone()
        if not game_result:
            stats_skipped += 1
            reason = f'S{season}.G{game_str}'
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        game_id = game_result[0]
        
        # Get player_id
        cursor.execute('SELECT player_id FROM players WHERE player_name = ?', (player,))
        player_result = cursor.fetchone()
        if not player_result:
            stats_skipped += 1
            skip_reasons['player_not_found'] = skip_reasons.get('player_not_found', 0) + 1
            continue
        player_id = player_result[0]
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO player_game_stats (
                    game_id, player_id, captain,
                    completions, attempts, completion_ratio,
                    passing_yards, passing_tds, interceptions_thrown, qbr, qb_avg_drive,
                    rush_attempts, rush_yards, rush_tds,
                    receptions, receiving_yards, receiving_tds,
                    tackles, sacks, interceptions, int_tds,
                    fumble_recoveries, fumble_tds, kickoff_tds, safeties,
                    win_loss, offensive_fantasy, defensive_fantasy, total_fantasy,
                    player_game_label, streak, game_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id, player_id,
                safe_str(parts[captain_idx]),
                safe_int(parts[comps_idx]),
                safe_int(parts[atts_idx]),
                safe_float(parts[ratio_idx]),
                safe_int(parts[yards_idx]),
                safe_int(parts[tds_idx]),
                safe_int(parts[ints_idx]),
                safe_float(parts[qbr_idx]),
                safe_float(parts[qb_avg_idx]),
                safe_int(parts[rush_att_idx]),
                safe_int(parts[rush_yds_idx]),
                safe_int(parts[rush_tds_idx]),
                safe_int(parts[recs_idx]),
                safe_int(parts[rec_yds_idx]),
                safe_int(parts[rec_tds_idx]),
                safe_int(parts[tacks_idx]),
                safe_float(parts[sacks_idx]),
                safe_int(parts[ints2_idx]),
                safe_int(parts[int_td_idx]),
                safe_int(parts[fum_rec_idx]),
                safe_int(parts[fum_td_idx]),
                safe_int(parts[ko_td_idx]),
                safe_int(parts[safety_idx]),
                safe_str(parts[wl_idx]),
                safe_float(parts[off_fant_idx]),
                safe_float(parts[def_fant_idx]),
                safe_float(parts[fant_idx]),
                safe_str(parts[pg_label_idx]),
                safe_int(parts[streak_idx]),
                safe_int(parts[gc_idx])
            ))
            stats_loaded += 1
            
        except Exception as e:
            stats_skipped += 1
            skip_reasons['insert_error'] = skip_reasons.get('insert_error', 0) + 1
            if stats_skipped <= 3:
                print(f"   ⚠️  Line {line_num}: Error inserting - {e}")
            continue
    
    # Print skip reasons summary
    if skip_reasons:
        print(f"\n   Skip reasons (top 10):")
        sorted_reasons = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:10]:
            print(f"      {reason}: {count} records")
        if len(skip_reasons) > 10:
            print(f"      ... and {len(skip_reasons) - 10} more reasons")
    
    return stats_loaded, stats_skipped
