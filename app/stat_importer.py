"""
Stat Importer - Replace Excel-based game entry
"""
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import re


def parse_stat_code(code, player_codes):
    """
    Parse stat codes like 'TS17L' into components
    Returns: (QB, yards, receiver/rusher, tackler, play_type)
    """
    code = code.strip().upper()
    
    if not code:
        return None
    
    # Pattern: [QB Code][Receiver/Rusher Code][Yards][Tackler Code]
    # Examples: TS17L, Q13S, CL (kickoff)
    
    result = {
        'original': code,
        'qb': None,
        'rusher': None,
        'receiver': None,
        'tackler': None,
        'yards': 0,
        'detail': None
    }
    
    # Check for detail codes (K=Kickoff, I=Interception, S=Sack, P=Punt)
    detail_match = re.search(r'[KISP]', code)
    if detail_match:
        result['detail'] = detail_match.group()
    
    # Extract yards (one or more digits)
    yards_match = re.search(r'\d+', code)
    if yards_match:
        result['yards'] = int(yards_match.group())
    
    # Extract player codes (single letters)
    # First letter is usually QB or ball carrier
    if len(code) > 0:
        result['qb'] = player_codes.get(code[0], code[0])
    
    # Second letter (if exists and is a letter) is usually receiver/rusher
    if len(code) > 1 and code[1].isalpha() and code[1] not in 'KISP':
        result['receiver'] = player_codes.get(code[1], code[1])
    
    # Last letter is usually tackler (after yards)
    letters_only = re.sub(r'[^A-Z]', '', code)
    if len(letters_only) > 1:
        result['tackler'] = player_codes.get(letters_only[-1], letters_only[-1])
    
    return result


def calculate_field_position(plays_df):
    """
    Calculate field position for each play (like the VBA code)
    """
    plays_df['start_yard'] = None
    plays_df['end_yard'] = None
    
    for i in range(len(plays_df) - 1, -1, -1):
        play = plays_df.iloc[i]
        
        # Skip turnovers, punts
        if play['is_fumble'] or play['is_interception'] or play['play_type'] == 'Punt':
            continue
        
        # Touchdown - set end position
        if play['is_touchdown']:
            if play['offense_team'] == 1:
                plays_df.at[i, 'end_yard'] = 60
                plays_df.at[i, 'start_yard'] = 60 - play['yards']
            else:
                plays_df.at[i, 'end_yard'] = 0
                plays_df.at[i, 'start_yard'] = 0 + play['yards']
        
        # Regular play - calculate based on previous play
        elif i < len(plays_df) - 1:
            next_start = plays_df.at[i + 1, 'start_yard']
            if next_start is not None:
                if play['offense_team'] == 1:
                    plays_df.at[i, 'end_yard'] = next_start
                    plays_df.at[i, 'start_yard'] = next_start - play['yards']
                else:
                    plays_df.at[i, 'end_yard'] = next_start
                    plays_df.at[i, 'start_yard'] = next_start + play['yards']
    
    return plays_df


def calculate_player_stats(plays_df, player_codes):
    """
    Calculate aggregated player stats from plays (replaces Excel formulas)
    """
    stats = {}
    
    for player_code, player_name in player_codes.items():
        stats[player_name] = {
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
            'int_tds': 0
        }
    
    # Process each play
    for _, play in plays_df.iterrows():
        # Passing stats
        if play['qb'] and play['qb'] in player_codes.values():
            stats[play['qb']]['attempts'] += 1
            if not play['is_incomplete']:
                stats[play['qb']]['completions'] += 1
                stats[play['qb']]['passing_yards'] += play['yards']
            if play['is_touchdown'] and play['receiver']:
                stats[play['qb']]['passing_tds'] += 1
            if play['is_interception']:
                stats[play['qb']]['interceptions_thrown'] += 1
        
        # Rushing stats
        if play['rusher'] and play['rusher'] in player_codes.values():
            stats[play['rusher']]['rush_attempts'] += 1
            stats[play['rusher']]['rush_yards'] += play['yards']
            if play['is_touchdown']:
                stats[play['rusher']]['rush_tds'] += 1
        
        # Receiving stats
        if play['receiver'] and play['receiver'] in player_codes.values():
            stats[play['receiver']]['receptions'] += 1
            stats[play['receiver']]['receiving_yards'] += play['yards']
            if play['is_touchdown']:
                stats[play['receiver']]['receiving_tds'] += 1
        
        # Defensive stats
        if play['tackler'] and play['tackler'] in player_codes.values():
            stats[play['tackler']]['tackles'] += 1
            if play['detail'] == 'S':  # Sack
                stats[play['tackler']]['sacks'] += 1
            if play['is_interception']:
                stats[play['tackler']]['interceptions'] += 1
                if play['is_touchdown']:
                    stats[play['tackler']]['int_tds'] += 1
    
    return stats
