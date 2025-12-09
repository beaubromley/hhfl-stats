"""
Verify and query the HHFL Stats database
"""
import sqlite3
import os


def verify_database(db_path='data/processed/hhfl_stats.db'):
    """Run verification queries on the database"""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("   Run create_database.py first!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*70)
    print("HHFL STATS DATABASE VERIFICATION & QUERIES")
    print("="*70)
    
    # Basic counts
    print("\n📊 DATABASE SUMMARY:")
    cursor.execute("SELECT COUNT(*) FROM games")
    print(f"   Total Games: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM players")
    print(f"   Total Players: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM player_game_stats")
    print(f"   Total Player Game Records: {cursor.fetchone()[0]}")
    
    # Top fantasy performers
    print("\n🏆 TOP 10 SINGLE-GAME FANTASY PERFORMANCES:")
    cursor.execute('''
        SELECT p.player_name, g.season, g.game_number, g.game_date,
               pgs.total_fantasy, pgs.offensive_fantasy, pgs.defensive_fantasy,
               g.captain1, g.captain2
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        JOIN games g ON pgs.game_id = g.game_id
        ORDER BY pgs.total_fantasy DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"   {i:2d}. {row[0]:20s} - S{row[1]}.G{row[2]} - {row[4]:.1f} pts "
              f"({row[5]:.1f} off, {row[6]:.1f} def)")
    
    # Career leaders
    print("\n📈 CAREER FANTASY POINTS LEADERS:")
    cursor.execute('''
        SELECT p.player_name,
               COUNT(*) as games,
               SUM(CASE WHEN win_loss = 'Win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN win_loss = 'Loss' THEN 1 ELSE 0 END) as losses,
               ROUND(SUM(total_fantasy), 1) as total_fantasy,
               ROUND(AVG(total_fantasy), 1) as avg_fantasy
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        WHERE pgs.total_fantasy > 0
        GROUP BY p.player_id
        ORDER BY total_fantasy DESC
        LIMIT 15
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        win_pct = (row[2] / row[1] * 100) if row[1] > 0 else 0
        print(f"   {i:2d}. {row[0]:20s} - {row[4]:7.1f} pts in {row[1]:3d} games "
              f"({row[5]:.1f} avg) - {row[2]}W-{row[3]}L ({win_pct:.1f}%)")
    
    # Passing leaders
    print("\n🎯 CAREER PASSING YARDS LEADERS:")
    cursor.execute('''
        SELECT p.player_name,
               SUM(completions) as comps,
               SUM(attempts) as atts,
               ROUND(SUM(completions) * 100.0 / NULLIF(SUM(attempts), 0), 1) as comp_pct,
               SUM(passing_yards) as yards,
               SUM(passing_tds) as tds,
               SUM(interceptions_thrown) as ints
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        WHERE attempts > 0
        GROUP BY p.player_id
        ORDER BY yards DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"   {i:2d}. {row[0]:20s} - {row[4]:5d} yds, {row[1]}/{row[2]} ({row[3]:.1f}%), "
              f"{row[5]} TDs, {row[6]} INTs")
    
    # Receiving leaders
    print("\n🏈 CAREER RECEIVING YARDS LEADERS:")
    cursor.execute('''
        SELECT p.player_name,
               SUM(receptions) as recs,
               SUM(receiving_yards) as yards,
               SUM(receiving_tds) as tds,
               ROUND(SUM(receiving_yards) * 1.0 / NULLIF(SUM(receptions), 0), 1) as ypr
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        WHERE receptions > 0
        GROUP BY p.player_id
        ORDER BY yards DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"   {i:2d}. {row[0]:20s} - {row[2]:5d} yds, {row[1]:4d} recs, "
              f"{row[3]:3d} TDs ({row[4]:.1f} ypr)")
    
    # Defensive leaders
    print("\n🛡️  CAREER TACKLES LEADERS:")
    cursor.execute('''
        SELECT p.player_name,
               SUM(tackles) as tackles,
               SUM(sacks) as sacks,
               SUM(interceptions) as ints,
               SUM(defensive_fantasy) as def_fantasy
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        WHERE tackles > 0 OR sacks > 0 OR interceptions > 0
        GROUP BY p.player_id
        ORDER BY tackles DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"   {i:2d}. {row[0]:20s} - {row[1]:4d} tackles, {row[2]:.1f} sacks, "
              f"{row[3]:3d} INTs ({row[4]:.1f} def pts)")
    
    # Season breakdown
    print("\n📅 SEASON SUMMARY:")
    cursor.execute('''
        SELECT g.season,
               COUNT(DISTINCT g.game_id) as games,
               COUNT(DISTINCT pgs.player_id) as unique_players,
               ROUND(AVG(g.team1_score + g.team2_score), 1) as avg_total_score,
               SUM(CASE WHEN g.overtime = 'Yes' THEN 1 ELSE 0 END) as ot_games
        FROM games g
        LEFT JOIN player_game_stats pgs ON g.game_id = pgs.game_id
        WHERE g.season IS NOT NULL
        GROUP BY g.season
        ORDER BY g.season
    ''')
    
    for row in cursor.fetchall():
        print(f"   Season {row[0]:2d}: {row[1]:3d} games, {row[2]:3d} players, "
              f"Avg Score: {row[3]:.1f}, OT Games: {row[4]}")
    
    # Head-to-head records
    print("\n⚔️  CAPTAIN HEAD-TO-HEAD RECORDS (Top Rivalries):")
    cursor.execute('''
        SELECT captain1, captain2,
               COUNT(*) as games,
               SUM(CASE WHEN team1_score > team2_score THEN 1 ELSE 0 END) as c1_wins,
               SUM(CASE WHEN team2_score > team1_score THEN 1 ELSE 0 END) as c2_wins,
               SUM(CASE WHEN team1_score = team2_score THEN 1 ELSE 0 END) as ties
        FROM games
        WHERE captain1 IS NOT NULL AND captain2 IS NOT NULL
        GROUP BY captain1, captain2
        HAVING games > 5
        ORDER BY games DESC
        LIMIT 10
    ''')
    
    for row in cursor.fetchall():
        total_games = row[2]
        c1_wins = row[3]
        c2_wins = row[4]
        ties = row[5]
        c1_pct = (c1_wins / total_games * 100) if total_games > 0 else 0
        print(f"   {row[0]:18s} vs {row[1]:18s}: {total_games:3d} games "
              f"({c1_wins:2d}-{c2_wins:2d}-{ties}, {c1_pct:.1f}%)")
    
    # MVP awards
    print("\n🏅 MOST MVP AWARDS:")
    cursor.execute('''
        SELECT mvp, COUNT(*) as mvp_count
        FROM games
        WHERE mvp IS NOT NULL AND mvp != ''
        GROUP BY mvp
        ORDER BY mvp_count DESC
        LIMIT 10
    ''')
    
    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"   {i:2d}. {row[0]:20s} - {row[1]:3d} MVPs")
    
    conn.close()
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    verify_database()
