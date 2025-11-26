import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
from mplsoccer import Pitch, VerticalPitch, Sbopen, FontManager

# Funktion: Berechne euklidische Distanz zwischen zwei Punkten
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def blocker_distance(row):
    if pd.isna(row["xBlo"]) or pd.isna(row["yBlo"]):
        return np.nan # Gibt nan zurück, wenn x oder y leer sind
    shooter = (row["xFe"], row["yFe"])
    blocker = (row["xBlo"], row["yBlo"])
    return distance(shooter, blocker)

def stresser_distance(row):
    """
    Berechnet die Distanz zwischen Schütze (xFe, yFe) und Stresser (xStr, yStr)
    für eine Zeile des DataFrames.
    """
    if pd.isna(row["xStr"]) or pd.isna(row["yStr"]):
        return np.nan # Gibt nan zurück, wenn x oder y leer sind
    shooter = (row["xFe"], row["yFe"])
    stresser = (row["xStr"], row["yStr"])
    return distance(shooter, stresser)

def goal_distance(xFe, yFe):
    # Pfostenkoordinaten
    x_left = 3.66 
    y_goal = 52.5
    x_right = -3.66

    if -3.66 <= xFe <= 3.66:
        # Zwischen den Pfosten → Lot auf die Torlinie
        return abs(yFe - y_goal)
    elif xFe > 3.66:
        # Links → Distanz zum linken Pfosten
        return np.sqrt((xFe - x_left)**2 + (yFe - y_goal)**2)
    elif xFe < -3.66:
        # Rechts → Distanz zum rechten Pfosten
        return np.sqrt((xFe - x_right)**2 + (yFe - y_goal)**2)

def goal_angle(xFe, yFe):
    # Pfostenkoordinaten
    left_post = np.array([3.66, 52.5])
    right_post = np.array([-3.66, 52.5])
    shooter = np.array([xFe, yFe])
    
    # Vektoren vom Schützen zu den Pfosten
    vec_left = left_post - shooter
    vec_right = right_post - shooter

    # Winkel zwischen beiden Vektoren berechnen
    dot = np.dot(vec_left, vec_right)
    norm_left = np.linalg.norm(vec_left)
    norm_right = np.linalg.norm(vec_right)

    # Absicherung: Division durch 0 verhindern
    if norm_left == 0 or norm_right == 0:
        return np.nan
    
    # Kosinus des Winkels (zwischen beiden Pfostenlinien)
    cos_theta = dot / (norm_left * norm_right)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # numerische Absicherung
    angle_rad = np.arccos(cos_theta)

    # Rückgabe in Grad
    return np.degrees(angle_rad)

# Funktion: Wendet das auf jede Zeile des DataFrames an
def keeper_distance(row):
    """
    Berechnet die Distanz zwischen Schütze (xFe, yFe) und TW (xTW, yTW)
    für eine Zeile des DataFrames.
    """
    shooter = (row["xFe"], row["yFe"])
    keeper = (row["xTW"], row["yTW"])
    return distance(shooter, keeper)

# Teamnamen an Hand der TID finden
def find_club(df, team):
    for i in range(len(df["TID"])):
        if df["TID"].iloc[i]==team:
            return df["Vereinsname"].iloc[i]

# Teamnfarbe an Hand der TID finden
def find_color(df, team):
    for i in range(len(df["TID"])):
        if df["TID"].iloc[i]==team:
            return df["color"].iloc[i]