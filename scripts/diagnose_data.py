"""
Diagnose missing data and mismatches
"""
import sqlite3

def diagnose_database(db_path='data/processed/hhfl_stats.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*70)
    print("DATA DIAGNOSTIC REPORT")
    print("="*70)
    
    # Check for duplicate games
    print("\n🔍 Checking for duplicate games...")
    cursor.execute('''
        SELECT season, game_number, COUNT(*) as count
        FROM games
        GROUP BY season, game_number
        HAVING count > 1
        ORDER BY season, game_number
    ''')
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"   ⚠️  Found {len(duplicates)} duplicate game entries:")
        for dup in duplicates[:10]:
            print(f"      Season {dup[0]}, Game {dup[1]}: {dup[2]} entries")
    else:
        print("   ✅ No duplicate games found")
    
    # Check which games have no stats
    print("\n🔍 Games with no player stats:")
    cursor.execute('''
        SELECT g.season, g.game_number, g.game_date, g.captain1, g.captain2
        FROM games g
        LEFT JOIN player_game_stats pgs ON g.game_id = pgs.game_id
        WHERE pgs.stat_id IS NULL
        ORDER BY g.season, g.game_number
        LIMIT 20
    ''')
    
    no_stats = cursor.fetchall()
    if no_stats:
        print(f"   ⚠️  Found {len(no_stats)} games with no stats (showing first 20):")
        for game in no_stats:
            print(f"      S{game[0]}.G{game[1]} - {game[2]} - {game[3]} vs {game[4]}")
    else:
        print("   ✅ All games have player stats")
    
    # Check season/game combinations in games table
    print("\n📊 Games per season:")
    cursor.execute('''
        SELECT season, COUNT(*) as game_count,
               MIN(game_number) as min_game,
               MAX(game_number) as max_game
        FROM games
        WHERE season IS NOT NULL
        GROUP BY season
        ORDER BY season
    ''')
    
    for row in cursor.fetchall():
        print(f"   Season {row[0]:2d}: {row[1]:3d} games (Game #{row[2]} to #{row[3]})")
    
    # Check for stats without matching games
    print("\n🔍 Checking stats file for unique season/game combinations...")
    
    # Read the stats file directly
    try:
        with open('data/raw/player_stats.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        header = lines[0].strip().split('\t')
        s_idx = header.index('S') if 'S' in header else 0
        g_idx = header.index('G') if 'G' in header else 1
        
        season_game_combos = set()
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) > max(s_idx, g_idx):
                try:
                    season = int(parts[s_idx]) if parts[s_idx].strip() else None
                    game = int(parts[g_idx]) if parts[g_idx].strip() else None
                    if season and game:
                        season_game_combos.add((season, game))
                except:
                    pass
        
        print(f"   📋 Found {len(season_game_combos)} unique season/game combinations in stats file")
        
        # Check which ones don't exist in games table
        missing_games = []
        for season, game in sorted(season_game_combos):
            cursor.execute('SELECT game_id FROM games WHERE season = ? AND game_number = ?', 
                          (season, game))
            if not cursor.fetchone():
                missing_games.append((season, game))
        
        if missing_games:
            print(f"\n   ⚠️  Found {len(missing_games)} season/game combos in stats but NOT in games:")
            for s, g in missing_games[:20]:
                print(f"      Season {s}, Game {g}")
            if len(missing_games) > 20:
                print(f"      ... and {len(missing_games) - 20} more")
        else:
            print("   ✅ All stats reference valid games")
            
    except Exception as e:
        print(f"   ❌ Error reading stats file: {e}")
    
    # Player with most games
    print("\n👥 Most Active Players:")
    cursor.execute('''
        SELECT p.player_name, COUNT(*) as games_played,
               SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN win_loss = 'Loss' THEN 1 ELSE 0 END) as losses
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        GROUP BY p.player_id
        ORDER BY games_played DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        win_pct = (row[2] / row[1] * 100) if row[1] > 0 else 0
        print(f"   {i:2d}. {row[0]:20s} - {row[1]:3d} games ({row[2]}-{row[3]}, {win_pct:.1f}%)")
    
    conn.close()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    diagnose_database()
