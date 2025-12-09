"""
Debug the data parser to see what's happening
"""

def debug_parser():
    print("="*70)
    print("DATA PARSER DEBUG")
    print("="*70)
    
    # Check games file
    print("\n📋 Analyzing GAMES file...")
    with open('data/raw/games.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split('\t')
    print(f"\nHeader columns ({len(header)} total):")
    for i, col in enumerate(header[:15]):
        print(f"   {i:2d}: '{col}'")
    
    # Show first 5 data rows
    print(f"\nFirst 5 data rows:")
    for i in range(1, min(6, len(lines))):
        parts = lines[i].strip().split('\t')
        print(f"\n   Row {i}: {len(parts)} columns")
        for j in range(min(10, len(parts))):
            print(f"      Col {j} ({header[j] if j < len(header) else 'unknown'}): '{parts[j]}'")
    
    # Check for rows where Season or Game are blank
    print("\n🔍 Checking for blank Season/Game values...")
    try:
        season_idx = header.index('Season')
        game_idx = header.index('Game')
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        print(f"   Available columns: {header}")
        return
    
    blank_count = 0
    zero_count = 0
    valid_count = 0
    
    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split('\t')
        if len(parts) <= max(season_idx, game_idx):
            continue
            
        season = parts[season_idx].strip()
        game = parts[game_idx].strip()
        
        if not season or not game:
            blank_count += 1
            if blank_count <= 3:
                print(f"   Row {i}: Season='{season}', Game='{game}'")
        elif game == '0':
            zero_count += 1
            if zero_count <= 3:
                print(f"   Row {i}: Has Game=0 - {parts[:8]}")
        else:
            valid_count += 1
    
    print(f"\n   Summary:")
    print(f"   ✅ Valid: {valid_count}")
    print(f"   ⚠️  Blank: {blank_count}")
    print(f"   ⚠️  Zero: {zero_count}")
    
    # Now check stats file
    print("\n" + "="*70)
    print("\n📋 Analyzing PLAYER STATS file...")
    
    with open('data/raw/player_stats.txt', 'r', encoding='utf-8') as f:
        stats_lines = f.readlines()
    
    stats_header = stats_lines[0].strip().split('\t')
    print(f"\nHeader columns ({len(stats_header)} total):")
    for i, col in enumerate(stats_header[:20]):
        print(f"   {i:2d}: '{col}'")
    
    # Show first 3 data rows
    print(f"\nFirst 3 data rows:")
    for i in range(1, min(4, len(stats_lines))):
        parts = stats_lines[i].strip().split('\t')
        print(f"\n   Row {i}: {len(parts)} columns")
        for j in range(min(12, len(parts))):
            col_name = stats_header[j] if j < len(stats_header) else 'unknown'
            print(f"      Col {j} ({col_name}): '{parts[j]}'")
    
    # Check for mismatches
    try:
        s_idx = stats_header.index('S')
        g_idx = stats_header.index('G')
        player_idx = stats_header.index('Player')
    except ValueError as e:
        print(f"   ❌ Error finding columns: {e}")
        return
    
    print("\n🔍 Checking season/game combinations in stats...")
    
    stats_season_games = {}
    blank_season = 0
    blank_game = 0
    blank_player = 0
    
    for i, line in enumerate(stats_lines[1:], start=2):
        parts = line.strip().split('\t')
        if len(parts) <= max(s_idx, g_idx, player_idx):
            continue
        
        season = parts[s_idx].strip()
        game = parts[g_idx].strip()
        player = parts[player_idx].strip()
        
        if not season:
            blank_season += 1
        if not game:
            blank_game += 1
        if not player:
            blank_player += 1
            
        if season and game:
            key = f"S{season}.G{game}"
            if key not in stats_season_games:
                stats_season_games[key] = 0
            stats_season_games[key] += 1
    
    print(f"\n   Found {len(stats_season_games)} unique season/game combos")
    print(f"   ⚠️  Blank seasons: {blank_season}")
    print(f"   ⚠️  Blank games: {blank_game}")
    print(f"   ⚠️  Blank players: {blank_player}")
    
    # Show combos with most records
    print("\n   Top 10 games by record count:")
    sorted_games = sorted(stats_season_games.items(), key=lambda x: x[1], reverse=True)
    for i, (game, count) in enumerate(sorted_games[:10], 1):
        print(f"      {i:2d}. {game}: {count} player records")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)
    
    print("\n💡 Next step: Check if your games.txt has the correct format")
    print("   The issue is likely that games with blank dates are being loaded as Game #0")


if __name__ == "__main__":
    debug_parser()
