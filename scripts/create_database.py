"""
Main script to create HHFL Stats database from Excel/text files
"""
import sqlite3
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.load_data import (
    create_schema, 
    load_games_from_file, 
    load_player_stats_from_file
)


def create_hhfl_database(
    games_file='data/raw/games.txt',
    stats_file='data/raw/player_stats.txt',
    db_file='data/processed/hhfl_stats.db'
):
    """
    Create the complete HHFL database from source files
    
    Args:
        games_file: Path to games data file
        stats_file: Path to player stats data file
        db_file: Path where database should be created
    """
    
    print("="*70)
    print("HHFL STATS DATABASE CREATOR")
    print("="*70)
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Check if source files exist
    if not os.path.exists(games_file):
        print(f"❌ ERROR: Games file not found: {games_file}")
        print(f"   Please create this file and paste your games data into it.")
        return False
    
    if not os.path.exists(stats_file):
        print(f"❌ ERROR: Stats file not found: {stats_file}")
        print(f"   Please create this file and paste your player stats data into it.")
        return False
    
    # Connect to database
    print(f"\n📁 Creating database: {db_file}")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Step 1: Create schema
        print("\n🔨 Step 1: Creating database schema...")
        create_schema(cursor)
        conn.commit()
        
        # Step 2: Load games
        print("\n📥 Step 2: Loading games data...")
        games_loaded = load_games_from_file(games_file, cursor)
        conn.commit()
        print(f"   ✅ Loaded {games_loaded} games")
        
        # Step 3: Load player stats
        print("\n📥 Step 3: Loading player stats...")
        stats_loaded, stats_skipped = load_player_stats_from_file(stats_file, cursor)
        conn.commit()
        print(f"   ✅ Loaded {stats_loaded} player game records")
        if stats_skipped > 0:
            print(f"   ⚠️  Skipped {stats_skipped} records (missing game/player)")
        
        # Step 4: Verification
        print("\n🔍 Step 4: Verifying database...")
        
        cursor.execute("SELECT COUNT(*) FROM games")
        game_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM players")
        player_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_game_stats")
        stat_count = cursor.fetchone()[0]
        
        print(f"   📊 Games: {game_count}")
        print(f"   👥 Players: {player_count}")
        print(f"   📈 Player Game Stats: {stat_count}")
        
        # Display sample data
        print("\n" + "="*70)
        print("SAMPLE DATA VERIFICATION")
        print("="*70)
        
        print("\n📅 Recent Games:")
        cursor.execute('''
            SELECT game_date, season, game_number, captain1, captain2, 
                   team1_score, team2_score, mvp
            FROM games
            ORDER BY season DESC, game_number DESC
            LIMIT 5
        ''')
        
        for row in cursor.fetchall():
            print(f"   S{row[1]}.G{row[2]:2d} - {row[0]} - {row[3]} vs {row[4]} "
                  f"({row[5]}-{row[6]}) MVP: {row[7] or 'N/A'}")
        
        print("\n🏆 Top 5 All-Time Fantasy Performances:")
        cursor.execute('''
            SELECT p.player_name, g.season, g.game_number, g.game_date,
                   pgs.total_fantasy, pgs.offensive_fantasy, pgs.defensive_fantasy
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.total_fantasy > 0
            ORDER BY pgs.total_fantasy DESC
            LIMIT 5
        ''')
        
        for row in cursor.fetchall():
            print(f"   {row[0]:20s} S{row[1]}.G{row[2]} ({row[3]}) - "
                  f"{row[4]:.1f} pts ({row[5]:.1f} off, {row[6]:.1f} def)")
        
        print("\n" + "="*70)
        print("✅ DATABASE CREATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Database saved to: {db_file}")
        print(f"💾 Database size: {os.path.getsize(db_file) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        conn.close()


if __name__ == "__main__":
    # Run the database creation
    success = create_hhfl_database()
    
    if success:
        print("\n🎉 You can now use verify_database.py to run queries!")
    else:
        print("\n❌ Database creation failed. Please check the errors above.")
