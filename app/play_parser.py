"""
Play Parser - Convert shorthand codes to structured play data
Replicates Excel formula logic in Python
"""
import re


class PlayParser:
    """Parse HHFL play codes and calculate stats"""
    
    def __init__(self, player_codes):
        """
        Initialize with player code mappings
        player_codes: dict like {'Q': 'Jimmy Quinn', 'T': 'Tyler Smith', ...}
        """
        self.player_codes = player_codes
        # Reverse lookup: player name to code
        self.name_to_code = {v: k for k, v in player_codes.items()}
    
    def parse_play(self, stat_code, detail_code=''):
        """
        Parse a play code like 'TS17L' or 'Q' or 'CL'
        
        Returns dict with:
        - qb, rusher, receiver, tackler, returner (player names)
        - yards, play_type, is_touchdown, is_incomplete, is_interception, etc.
        """
        stat_code = stat_code.strip().upper()
        detail_code = detail_code.strip().upper()
        
        play = {
            'stat_code': stat_code,
            'detail_code': detail_code,
            'qb': None,
            'rusher': None,
            'receiver': None,
            'tackler': None,
            'returner': None,
            'fumble_recovery': None,
            'yards': 0,
            'play_type': None,
            'is_touchdown': False,
            'is_incomplete': False,
            'is_fumble': False,
            'is_interception': False,
            'is_safety': False,
            'is_turnover': False,
            'description': ''
        }
        
        # Empty play
        if not stat_code:
            return play
        
        # Extract yards from stat code (any digits)
        yards_match = re.search(r'\d+', stat_code)
        if yards_match:
            play['yards'] = int(yards_match.group())
        
        # Determine play type based on detail code
        if detail_code == 'K':
            play['play_type'] = 'Kickoff'
            play['is_touchdown'] = 'KT' in detail_code or 'T' in stat_code
        elif detail_code == 'KT':
            play['play_type'] = 'Kickoff TD'
            play['is_touchdown'] = True
        elif detail_code == 'I':
            play['play_type'] = 'INT'
            play['is_interception'] = True
            play['is_turnover'] = True
        elif detail_code == 'P':
            play['play_type'] = 'Punt'
        elif detail_code == 'S':
            play['play_type'] = 'Sack'
        elif detail_code == 'F':
            play['play_type'] = 'Fumble'
            play['is_fumble'] = True
            play['is_turnover'] = True
        elif not stat_code or len(stat_code) == 1:
            # Single letter = incomplete pass
            play['play_type'] = 'Incomplete'
            play['is_incomplete'] = True
        
        # Parse player codes from stat_code
        # Remove all numbers and detail codes to get just player codes
        player_letters = re.sub(r'[0-9]', '', stat_code)
        
        # Determine if pass or rush based on code length and yards position
        if play['yards'] > 0 and len(player_letters) >= 2:
            # Check if yards are in middle (pass) or at end (rush)
            yards_pos = stat_code.find(str(play['yards']))
            
            # If there's a letter after the yards, it's a pass (receiver before yards, tackler after)
            if yards_pos > 0 and yards_pos < len(stat_code) - len(str(play['yards'])):
                # Pass play: QB + Receiver + Yards + Tackler
                play['play_type'] = 'Pass TD' if 'T' in detail_code or stat_code.endswith('T') else 'Pass'
                
                if len(player_letters) >= 1:
                    play['qb'] = self.player_codes.get(player_letters[0])
                if len(player_letters) >= 2:
                    play['receiver'] = self.player_codes.get(player_letters[1])
                if len(player_letters) >= 3:
                    play['tackler'] = self.player_codes.get(player_letters[2])
                
                # Check for TD
                if stat_code.endswith('T') and len(stat_code) == len(player_letters) + len(str(play['yards'])) + 1:
                    # Last character is T and it's part of stat_code length
                    play['is_touchdown'] = True
                    play['play_type'] = 'Pass TD'
                    # Remove tackler since T means touchdown
                    play['tackler'] = None
            else:
                # Rush play: QB/Rusher + Yards + Tackler
                play['play_type'] = 'Rush TD' if 'T' in detail_code else 'Rush'
                
                if len(player_letters) >= 1:
                    play['rusher'] = self.player_codes.get(player_letters[0])
                    play['qb'] = None  # Rusher is also the QB in this case
                if len(player_letters) >= 2:
                    play['tackler'] = self.player_codes.get(player_letters[1])
        
        elif len(player_letters) == 1:
            # Single letter with no yards = incomplete pass
            play['qb'] = self.player_codes.get(player_letters[0])
            play['play_type'] = 'Incomplete'
            play['is_incomplete'] = True
        
        elif len(player_letters) >= 2 and play['yards'] == 0:
            # Two letters, no yards = special play (kickoff, sack, etc.)
            if detail_code == 'K':
                play['returner'] = self.player_codes.get(player_letters[1])
                play['tackler'] = self.player_codes.get(player_letters[0])
            elif detail_code == 'S':
                play['qb'] = self.player_codes.get(player_letters[0])
                play['tackler'] = self.player_codes.get(player_letters[1])
        
        # Special case for interceptions with players
        if detail_code == 'I' and len(player_letters) >= 2:
            play['qb'] = self.player_codes.get(player_letters[0])
            play['receiver'] = self.player_codes.get(player_letters[1])  # Interceptor
            if len(player_letters) >= 3:
                play['tackler'] = self.player_codes.get(player_letters[2])
        
        # Generate description
        play['description'] = self._generate_description(play)
        
        return play
    
    def _generate_description(self, play):
        """Generate human-readable play description (like Excel formula)"""
        desc = ""
        
        if play['play_type'] == 'Kickoff':
            if play['returner']:
                desc = f"Kickoff returned by {play['returner']}"
                if play['tackler']:
                    desc += f" and tackled by {play['tackler']}"
            else:
                desc = f"Kickoff, tackled by {play['tackler']}" if play['tackler'] else "Kickoff"
        
        elif play['play_type'] == 'Kickoff TD':
            desc = f"Kickoff returned for touchdown by {play['returner']}" if play['returner'] else "Kickoff TD"
        
        elif play['play_type'] == 'Pass':
            desc = f"{play['qb']} pass complete to {play['receiver']} for {play['yards']} yards"
            if play['tackler']:
                desc += f" tackled by {play['tackler']}"
        
        elif play['play_type'] == 'Pass TD':
            desc = f"{play['qb']} pass complete to {play['receiver']} for {play['yards']} yards and a TD"
        
        elif play['play_type'] == 'Rush':
            rusher = play['rusher'] or play['qb']
            desc = f"{rusher} run for {play['yards']} yards"
            if play['tackler']:
                desc += f" tackled by {play['tackler']}"
        
        elif play['play_type'] == 'Rush TD':
            rusher = play['rusher'] or play['qb']
            desc = f"{rusher} run for {play['yards']} yards and a TD"
        
        elif play['play_type'] == 'Incomplete':
            desc = f"{play['qb']} incomplete pass" if play['qb'] else "Incomplete pass"
        
        elif play['play_type'] == 'INT':
            desc = f"{play['qb']} pass intercepted by {play['receiver']}" if play['qb'] and play['receiver'] else "Interception"
            if play['tackler']:
                desc += f" tackled by {play['tackler']}"
        
        elif play['play_type'] == 'Sack':
            desc = f"{play['qb']} sacked by {play['tackler']}" if play['qb'] and play['tackler'] else "Sack"
        
        elif play['play_type'] == 'Punt':
            desc = "Punt"
        
        elif play['play_type'] == 'Fumble':
            desc = f"Fumble recovered by {play['fumble_recovery']}" if play['fumble_recovery'] else "Fumble"
        
        else:
            desc = f"Play: {play['stat_code']}"
        
        return desc
    
    def calculate_game_stats(self, plays):
        """
        Calculate all player stats from plays list
        Returns: (player_stats_dict, team1_score, team2_score)
        """
        stats = {}
        team1_score = 0
        team2_score = 0
        
        # Initialize stats for all players
        for code, name in self.player_codes.items():
            stats[name] = {
                'completions': 0,
                'attempts': 0,
                'passing_yards': 0,
                'passing_tds': 0,
                'interceptions_thrown': 0,
                'rush_attempts': 0,
                'rush_yards': 0,
                'rush_tds': 0,
                'receptions': 0,
                'receiving_yards': 0,
                'receiving_tds': 0,
                'tackles': 0,
                'sacks': 0,
                'interceptions': 0,
                'int_tds': 0,
                'fumble_recoveries': 0,
                'kickoff_returns': 0,
                'kickoff_tds': 0
            }
        
        current_offense = 1  # Start with team 1
        
        # Process each play
        for play in plays:
            # Passing stats
            if play['qb']:
                if play['play_type'] in ['Pass', 'Pass TD', 'Incomplete', 'INT']:
                    stats[play['qb']]['attempts'] += 1
                    
                    if play['play_type'] in ['Pass', 'Pass TD']:
                        stats[play['qb']]['completions'] += 1
                        stats[play['qb']]['passing_yards'] += play['yards']
                    
                    if play['play_type'] == 'Pass TD':
                        stats[play['qb']]['passing_tds'] += 1
                    
                    if play['is_interception']:
                        stats[play['qb']]['interceptions_thrown'] += 1
            
            # Rushing stats
            if play['rusher']:
                stats[play['rusher']]['rush_attempts'] += 1
                stats[play['rusher']]['rush_yards'] += play['yards']
                if play['play_type'] == 'Rush TD':
                    stats[play['rusher']]['rush_tds'] += 1
            
            # Receiving stats
            if play['receiver'] and not play['is_interception']:
                stats[play['receiver']]['receptions'] += 1
                stats[play['receiver']]['receiving_yards'] += play['yards']
                if play['play_type'] == 'Pass TD':
                    stats[play['receiver']]['receiving_tds'] += 1
            
            # Defensive stats
            if play['tackler']:
                stats[play['tackler']]['tackles'] += 1
                
                if play['play_type'] == 'Sack':
                    stats[play['tackler']]['sacks'] += 1
                
                if play['is_interception']:
                    stats[play['tackler']]['interceptions'] += 1
                    if play['is_touchdown']:
                        stats[play['tackler']]['int_tds'] += 1
            
            # Score tracking
            if play['is_touchdown']:
                if current_offense == 1:
                    team1_score += 1
                else:
                    team2_score += 1
            
            # Check for turnover (switches possession)
            if play['is_turnover'] or play['play_type'] == 'Punt':
                current_offense = 2 if current_offense == 1 else 1
        
        return stats, team1_score, team2_score
    
    def calculate_fantasy_points(self, stats_dict):
        """
        Calculate fantasy points (you'll need to provide your scoring rules)
        For now using standard fantasy scoring
        """
        offense = 0
        defense = 0
        
        # Offensive points
        offense += stats_dict['passing_yards'] * 0.04  # 0.04 per yard
        offense += stats_dict['passing_tds'] * 4  # 4 per TD
        offense -= stats_dict['interceptions_thrown'] * 2  # -2 per INT
        
        offense += stats_dict['rush_yards'] * 0.1  # 0.1 per yard
        offense += stats_dict['rush_tds'] * 6  # 6 per TD
        
        offense += stats_dict['receptions'] * 1  # 1 per catch (PPR)
        offense += stats_dict['receiving_yards'] * 0.1  # 0.1 per yard
        offense += stats_dict['receiving_tds'] * 6  # 6 per TD
        
        # Defensive points
        defense += stats_dict['tackles'] * 0.5  # 0.5 per tackle
        defense += stats_dict['sacks'] * 1  # 1 per sack
        defense += stats_dict['interceptions'] * 2  # 2 per INT
        defense += stats_dict['int_tds'] * 6  # 6 per pick-six
        
        total = offense + defense
        
        return round(offense, 2), round(defense, 2), round(total, 2)
