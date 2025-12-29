import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from mplsoccer import Pitch, VerticalPitch, Sbopen, FontManager

import football_functions_2 as ff

from scipy.stats import poisson

st.title("1. FC Nürnberg U16")
st.subheader("Expected Goals 2025/26")

# ================================================== ALLGEMEINE VORBEREITUNGEN ==================================================
# Neue Daten einlesen
@st.cache_data
def load_data():
    abschlüsse = pd.read_csv("abschlüsse_xG_2.1.csv")
    xls = "xG_U16_Anwendung.xlsx"
    teams = pd.read_excel(xls, sheet_name="Teams")
    spiele = pd.read_excel(xls, sheet_name="Spiele")
    spieler = pd.read_excel(xls, sheet_name="Spieler")
    spielzeiten = pd.read_excel(xls, sheet_name="Spielzeiten")
    karten = pd.read_excel(xls, sheet_name="Rote Karten")
    return abschlüsse, teams, spiele, spieler, spielzeiten, karten

abschlüsse, teams, spiele, spieler, spielzeiten, karten = load_data()

# Setting custom font
@st.cache_resource
def load_font():
    return font_manager.FontProperties(fname="dfb-sans-web-bold.64bb507.ttf")

font_props = load_font()

# Teamfarben festlegen
teams["color"] = ["#AA1124", "#F8D615", "#CD1719", "#ED1248", "#006BB3", "#C20012", "#E3191B", "#03466A", 
                  "#2FA641", "#009C6B", "#ED1B24", "#E3000F", "#2E438C", "#5AAADF", "#EE232B"]

# Spielphasen ersetzen
abschlüsse["Spielphase"] = abschlüsse["Spielphase"].replace({
    "nFS": "Freistoss",
    "FSS": "Freistoss",
    "FSF": "Freistoss",
    "EB": "Eigener Ballbesitz",
    "EW": "Eigener Ballbesitz",
    "nEW": "Eigener Ballbesitz",
    "USO": "Umschalten Offensiv",
    "n11M": "Umschalten Offensiv",
    "E": "Ecke",
    "nE": "Ecke",
    "11M": "Elfmeter"})
abschlüsse.drop("Phase", axis=1, inplace=True)

# Körperteile umbenennen
abschlüsse["Körperteil"] = abschlüsse["Körperteil"].replace({
    "L": "Links",
    "R": "Rechts",
    "K": "Kopf",
    "S": "Sonstige"
})

# Vorbereitungen ersetzen
abschlüsse["Vorbereitung"] = abschlüsse["VorT"].replace({
    "TieferPass": "Tiefer Pass",
    "HoheFlanke": "Hohe Flanke",
    "FlacheHereingabe": "Flache Hereingabe",
    "Dribbling": "Dribbling",    
    "DirekterStandard": "Direkter Standard",
    "UnkontrollierteVorbereitung": "Unkontrollierte Vorbereitung",
    "KontrollierteVorlage": "Kontrollierte Vorlage",
})
abschlüsse.drop("VorT", axis=1, inplace=True)

# Spiele-Filter erstellen
spiele_filter = spiele[["SID", "Heim", "Gast"]].copy()
spiele_filter["Heimteam"] = spiele_filter["Heim"].map(teams.set_index("TID")["Vereinsname"])
spiele_filter["Gästeteam"] = spiele_filter["Gast"].map(teams.set_index("TID")["Vereinsname"])
spiele_filter["Spiel"] = "(" + spiele_filter["SID"].astype(str) + ") " + spiele_filter["Heimteam"] + ' - ' + spiele_filter["Gästeteam"]

# Selectbox Spiele
game_filter = st.selectbox("Spiel auswählen", spiele_filter["Spiel"].unique())
game = spiele_filter.loc[spiele_filter["Spiel"]==game_filter, "SID"].values[0]

# Spieler-Filter erstellen
spieler_filter = spieler.copy()
spieler_filter["Nachname"] = spieler_filter["Nachname"].str.replace(r"\s[A-Z]\.$", "", regex=True)
spieler_filter["Name"] = spieler_filter["Vorname"] + " " + spieler_filter["Nachname"]

# Datumsformat festlegen                  
from datetime import datetime
date = spiele.loc[spiele["SID"]==game, "Datum"].iloc[0]
date = date.strftime("%d.%m.%Y")

# ================================================== METRIK-VORBEREITUNGEN ==================================================
@st.cache_data
def compute_xgap_xpts(abschlüsse, spiele):
    # -------------------------------------------------- xGAP berechnen --------------------------------------------------
    xgap = abschlüsse.groupby(["SID", "AP"])["xG"].apply(lambda x: 1 - np.prod(1 - x))

    # max AbNr pro AP bestimmen
    max_abnr = abschlüsse.groupby(["SID", "AP"])["AbNr"].transform("max")

    # neue Spalte xGAP setzen (nur bei höchster AbNr einer AP)
    abschlüsse["xGAP"] = np.where(
        abschlüsse["AbNr"] == max_abnr,
        abschlüsse.set_index(["SID", "AP"]).index.map(xgap),
        np.nan
    )

    # -------------------------------------------------- xPoints berechnen --------------------------------------------------
    xG_end = abschlüsse.groupby(["SID", "HG"], as_index=False)["xGAP"].sum().rename(columns={"xGAP": "xG"})

    xg_h = xG_end[xG_end["HG"] == "H"].rename(columns={"xG": "xGH"})
    xg_a = xG_end[xG_end["HG"] == "G"].rename(columns={"xG": "xGG"})

    spiele_üb = spiele.copy()
    spiele_üb = (spiele_üb.merge(xg_h[["SID", "xGH"]], on="SID", how="left").merge(xg_a[["SID", "xGG"]], on="SID", how="left"))

    xPH_list = []
    xPG_list = []

    for _, row in spiele_üb.iterrows():
        lam_h = row["xGH"]
        lam_a = row["xGG"]

        h_goals_probs = [poisson.pmf(i, lam_h) for i in range(21)]
        a_goals_probs = [poisson.pmf(i, lam_a) for i in range(21)]

        match_probs = np.outer(h_goals_probs, a_goals_probs)

        p_h_win = np.sum(np.tril(match_probs, -1))
        p_h_draw = np.sum(np.diag(match_probs))
        p_h_loss = np.sum(np.triu(match_probs, 1))
        
        xp_h = (p_h_win * 3) + (p_h_draw * 1) + (p_h_loss * 0)
        xp_a = (p_h_win * 0) + (p_h_draw * 1) + (p_h_loss * 3)

        xPH_list.append(xp_h)
        xPG_list.append(xp_a)

    spiele_üb["xPH"] = xPH_list
    spiele_üb["xPG"] = xPG_list

    spiele_üb[['xGH', 'xGG', 'xPH', 'xPG']] = spiele_üb[['xGH', 'xGG', 'xPH', 'xPG']].round(2)

    heim = spiele_üb[['SID', 'Datum', 'Heim', 'TH', 'xGH', 'PH', 'xPH']].copy()
    gast = spiele_üb[['SID', 'Datum', 'Gast', 'TG', 'xGG', 'PG', 'xPG']].copy()

    heim["HG"] = "H"
    gast["HG"] = "G"

    heim = heim.rename(columns={"Heim": "Team", "TH": "Tore", "PH": "Punkte", "xGH": "xG", "xPH": "xPoints"})
    gast = gast.rename(columns={"Gast": "Team", "TG": "Tore", "PG": "Punkte", "xGG": "xG", "xPG": "xPoints"})

    spiele_üb = pd.concat([heim, gast], ignore_index=True)
    spiele_üb = spiele_üb[['SID', 'Datum', 'Team', 'HG', 'Tore', 'xG', 'Punkte', 'xPoints']].sort_values(["SID", "HG"], ascending=[True, False])

    heimspiele = heim[heim["Team"]=="FCN"]["SID"].tolist()
    auswärtsspiele = gast[gast["Team"]=="FCN"]["SID"].tolist()

    return abschlüsse, spiele_üb, heimspiele, auswärtsspiele

abschlüsse, spiele_üb, heimspiele, auswärtsspiele = compute_xgap_xpts(abschlüsse, spiele)

# ================================================== EINZELSPIEL ==================================================
# Heim- und Auswärtsteam splitten
df_h = abschlüsse[(abschlüsse["SID"] == game) & (abschlüsse["HG"] == "H")].copy()
df_a = abschlüsse[(abschlüsse["SID"] == game) & (abschlüsse["HG"] == "G")].copy()
# Auswärtskoordianten spiegeln
df_a[["yFe", "xFe"]] *= -1

# Fehlermeldung falls keine Abschlussdaten vorhanden
if df_h.empty and df_a.empty:
    st.error("Für dieses Spiel gibt es aktuell noch keine Daten!")
else:
    # Teamnamen und Farben
    home_team_id = df_h.TID.unique()[0]
    home_team = ff.find_club(teams, home_team_id)
    color_home = ff.find_color(teams, home_team_id)

    away_team_id = df_a.TID.unique()[0]
    away_team = ff.find_club(teams, away_team_id)
    color_away = ff.find_color(teams, away_team_id)

    # ---------- xGAP für xG-Timeline----------
    # Kumulierte xGAP
    df_h["cum_xGAP"] = df_h["xGAP"].cumsum().where(df_h["xGAP"].notna())
    df_a["cum_xGAP"] = df_a["xGAP"].cumsum().where(df_a["xGAP"].notna())

    # Split 1. und 2. Halbzeit
    df_h_1 = df_h[df_h["HZ"]==1]
    df_h_2 = df_h[df_h["HZ"]==2]

    df_a_1 = df_a[df_a["HZ"]==1]
    df_a_2 = df_a[df_a["HZ"]==2]

    # Einzelne cum_xGAP-Listen erstellen
    cum_xGAP_h_1 = [0]
    cum_xGAP_h_1.extend(df_h_1.loc[df_h_1["cum_xGAP"].notna(), "cum_xGAP"].tolist())
    max_cum_xGAP_h_1 = max(cum_xGAP_h_1)
    cum_xGAP_h_2 = [max_cum_xGAP_h_1]
    cum_xGAP_h_2.extend(df_h_2.loc[df_h_2["cum_xGAP"].notna(), "cum_xGAP"].tolist())

    cum_xGAP_a_1 = [0]
    cum_xGAP_a_1.extend(df_a_1.loc[df_a_1["cum_xGAP"].notna(), "cum_xGAP"].tolist())
    max_cum_xGAP_a_1 = max(cum_xGAP_a_1)
    cum_xGAP_a_2 = [max_cum_xGAP_a_1]
    cum_xGAP_a_2.extend(df_a_2.loc[df_a_2["cum_xGAP"].notna(), "cum_xGAP"].tolist())

    # Einzelne Listen der Minuten der xGAP-Zeitpunkte erstellen
    Min_h_1 = [0]
    Min_h_1.extend(df_h_1.loc[df_h_1["cum_xGAP"].notna(), "Min"].tolist())
    Min_h_2 = [40]
    Min_h_2.extend(df_h_2.loc[df_h_2["cum_xGAP"].notna(), "Min"].tolist())

    Min_a_1 = [0]
    Min_a_1.extend(df_a_1.loc[df_a_1["cum_xGAP"].notna(), "Min"].tolist())
    Min_a_2 = [40]
    Min_a_2.extend(df_a_2.loc[df_a_2["cum_xGAP"].notna(), "Min"].tolist())

    # Spielzeiten der Halbzeiten ermitteln
    end_1 = spiele.loc[spiele["SID"]==game, "1. HZ"].iloc[0]
    end_2 = spiele.loc[spiele["SID"]==game, "2. HZ"].iloc[0]

    # Letzte xGAP-Stufe bis zur letzten Minute (+1) durchziehen
    cum_xGAP_h_1.append(cum_xGAP_h_1[-1])
    Min_h_1.append(int(end_1+1))
    cum_xGAP_h_2.append(cum_xGAP_h_2[-1])
    Min_h_2.append(int(end_2+41))

    cum_xGAP_a_1.append(cum_xGAP_a_1[-1])
    Min_a_1.append(int(end_1+1))
    cum_xGAP_a_2.append(cum_xGAP_a_2[-1])
    Min_a_2.append(int(end_2+41))

    # Kummulierte höchste xGAP-Werte der zweiten Halbzeit ermitteln
    max_cum_xGAP_h_2 = max(cum_xGAP_h_2)
    max_cum_xGAP_a_2 = max(cum_xGAP_a_2)

    # Toranzahl ermitteln
    goals_h = int(df_h["Ergeb"].isin(["Tor", "ET"]).sum())
    goals_a = int(df_a["Ergeb"].isin(["Tor", "ET"]).sum())

    # Listen für die Tore des Heimteams
    goals_h_1 = []
    goals_h_1.extend(df_h_1.loc[df_h_1["Ergeb"].isin(["Tor", "ET"]), "cum_xGAP"].tolist())
    goals_h_2 = []
    goals_h_2.extend(df_h_2.loc[df_h_2["Ergeb"].isin(["Tor", "ET"]), "cum_xGAP"].tolist())
    gMin_h_1 = []
    gMin_h_1.extend(df_h_1.loc[df_h_1["Ergeb"].isin(["Tor", "ET"]), "Min"].tolist())
    gMin_h_2 = []
    gMin_h_2.extend(df_h_2.loc[df_h_2["Ergeb"].isin(["Tor", "ET"]), "Min"].tolist())

    # Listen für die Tore des Auswärtsteams
    goals_a_1 = []
    goals_a_1.extend(df_a_1.loc[df_a_1["Ergeb"].isin(["Tor", "ET"]), "cum_xGAP"].tolist())
    goals_a_2 = []
    goals_a_2.extend(df_a_2.loc[df_a_2["Ergeb"].isin(["Tor", "ET"]), "cum_xGAP"].tolist())
    gMin_a_1 = []
    gMin_a_1.extend(df_a_1.loc[df_a_1["Ergeb"].isin(["Tor", "ET"]), "Min"].tolist())
    gMin_a_2 = []
    gMin_a_2.extend(df_a_2.loc[df_a_2["Ergeb"].isin(["Tor", "ET"]), "Min"].tolist())

    # ---------- Vorbereitungen Spiel-Übersicht ----------
    df_goals = abschlüsse[(abschlüsse["SID"] == game) & abschlüsse["Ergeb"].isin(["Tor", "ET"])]

    df_goals = df_goals.merge(
        teams[["TID", "Vereinsname"]],
        how="left",
        left_on="TID",
        right_on="TID"
    )

    df_goals = df_goals.merge(
        spieler[["Nr", "Nachname"]],
        how="left",
        left_on="SNr",
        right_on="Nr"
    )
    df_goals = df_goals.rename(columns={"Nachname": "Schütze"})

    df_goals = df_goals.merge(
        spieler[["Nr", "Nachname"]],
        how="left",
        left_on="VNr",
        right_on="Nr"
    )
    df_goals = df_goals.rename(columns={"Nachname": "Vorlage"})

    df_goals = df_goals.rename(columns={"SpSt": "Ereignis"})
    df_goals = df_goals.rename(columns={"Schütze": "Spieler"})

    df_goals["Entstehung"] = "Regulär"
    df_goals.loc[df_goals["Spielphase"]=="Elfmeter", "Entstehung"] = "11M"
    df_goals.loc[df_goals["Ergeb"]=="ET", "Entstehung"] = "ET"

    df_goals = df_goals[["Min", "Ereignis", "Spieler", "xG", "Entstehung", "Vorlage", "Vereinsname"]].copy()

    df_karten = karten[karten["SID"]==game]

    df_karten = df_karten.merge(
        spieler[["Nr", "Nachname"]],
        how="left",
        left_on="TNr",
        right_on="Nr"
    )
    df_karten = df_karten.rename(columns={"Nachname": "Spieler"})

    df_karten = df_karten.merge(
        teams[["TID", "Vereinsname"]],
        how="left",
        left_on="Team",
        right_on="TID"
    )

    df_karten["Ereignis"] = "Rot"
    df_karten["xG"] = pd.NA
    df_karten["Entstehung"] = "Regulär"
    df_karten["Vorlage"] = pd.NA

    df_karten = df_karten[["Min", "Ereignis", "Spieler", "xG", "Entstehung", "Vorlage", "Vereinsname"]].copy()

    df_comb = pd.concat([df_goals, df_karten], ignore_index=True).sort_values("Min", ascending=True)

    liste = df_comb.apply(
        lambda x: (
            f"{x['Min']:02d}'"  # Minute zweistellig
            f"{"   " + x['Ereignis']}"
            f"{"   " + x["Vereinsname"] if x['Ereignis'] == "Rot" else ""}"
            f"{'   ' + x['Spieler'] if pd.notna(x['Spieler']) else ''}"  # Schütze (falls vorhanden)
            f"{', ' + x['Vorlage'] if pd.notna(x['Vorlage']) else ''}"  # Vorlage (falls vorhanden)
            f"{'   (' + x['Entstehung'] + ')' if x['Entstehung'] != 'Regulär' else ''}"  # Entstehung (nur wenn ≠ Regulär)
            f"{('   <1%' if 0 < x['xG'] < 0.01 else '   ' + format(x['xG']*100, '.0f') + '%') if pd.notna(x['xG']) and x['xG'] != 0 else ''}"
        ),
        axis=1
    )

    # ================================================== SPIEL-AUSGABE ==================================================
    #GridSpec initialisieren
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # -------------------------------------------------- GridSpec-Layout --------------------------------------------------
    background_color = "#262730" #"#F8F5E9"
    text_color = "white"

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.set_facecolor(background_color)

    gs = fig.add_gridspec(nrows = 3, ncols = 3)

    ax1 = fig.add_subplot(gs[0:2,2:3])
    ax2 = fig.add_subplot(gs[0:2,0:2])
    ax2.set_title("xG-Map", fontproperties = font_props, fontsize = 30, color=text_color)

    gs_tl = gs[2, 0:3].subgridspec(1, 2, wspace=0.0)
    ax_h1 = fig.add_subplot(gs_tl[0, 0]) # Timeline für erste Halbzeit
    ax_h2 = fig.add_subplot(gs_tl[0, 1], sharey=ax_h1) # Timeline für zweite Halbzeit mit gleicher y-Skalierung
    fig.text(0.5, 0.38, "xG-Timeline", ha="center", va="top", fontproperties=font_props, fontsize=30, color=text_color)

    # -------------------------------------------------- Spielverlauf --------------------------------------------------
    ax1.set_facecolor(background_color)
    ax1.axis("off")
    ax1.text(0.5, 0.91, "Spielverlauf", ha="center", va="center", fontproperties=font_props, fontsize=30, color=text_color)
    for i, txt in enumerate(liste):
        ax1.text(0.05, 0.85 - i * 0.05, txt, transform=ax1.transAxes, fontsize=14, fontproperties=font_props, va="top", ha="left", color=text_color)

    # -------------------------------------------------- xG-Map --------------------------------------------------
    # Spielfeld zeichnen
    pitch = Pitch(
        pitch_type='skillcorner', pitch_length=105, pitch_width=68,
        axis=False, label=False, tick=False,
        pad_left=3, pad_right=3, pad_top=3, pad_bottom=3,
        pitch_color=background_color, line_color=text_color,
        stripe=False, linewidth=1, corner_arcs=True, goal_type="box"
    )
    pitch.draw(ax=ax2)

    # Abschlüsse des Heimteams plotten
    for i in df_h.to_dict(orient="records"):
            pitch.scatter(
                i["yFe"],
                i["xFe"],
                marker = '*' if i["Ergeb"] == "Tor" else 'o',
                s = np.sqrt(i["xG"]) * 400 * (3 if i["Ergeb"] == "Tor" else 1),
                facecolors=to_rgba(color_home, 0.5),
                edgecolors=to_rgba(color_home, 1),
                linewidth = 1.5,
                zorder = 2,
                ax = ax2)

    # Abschlüsse des Auswärtsteams plotten
    for i in df_a.to_dict(orient="records"):
            pitch.scatter(
                i["yFe"],
                i["xFe"],
                marker = '*' if i["Ergeb"] == "Tor" else 'o',
                s = np.sqrt(i["xG"]) * 400 * (3 if i["Ergeb"] == "Tor" else 1),
                facecolors=to_rgba(color_away, 0.35),
                edgecolors=to_rgba(color_away, 1),
                linewidth = 1.2,
                zorder = 2,
                ax = ax2)

    # xPoints für Heim und Auswärts ermitteln
    xpts_h = spiele_üb.loc[((spiele_üb["SID"] == game) & (spiele_üb["HG"] == "H")), "xPoints"].values[0]
    xpts_a = spiele_üb.loc[((spiele_üb["SID"] == game) & (spiele_üb["HG"] == "G")), "xPoints"].values[0]

    team_size = 25; goal_size = 150; xg_size = 50; alpha = 0.5; xp_size = 30
    # Heim links
    ax2.text(-26.25, 27, home_team, fontproperties=font_props, color=color_home,
                ha='center', va='center', fontsize=team_size, alpha=alpha, zorder=1)
    ax2.text(-20, -3,   goals_h, fontproperties=font_props, color=color_home,
                ha='center', va='center', fontsize=goal_size, alpha=alpha, zorder=1)
    ax2.text(-20, -18, f"{max_cum_xGAP_h_2:.2f}", fontproperties=font_props, color=color_home,
                ha='center', va='center', fontsize=xg_size, alpha=alpha, zorder=1)
    ax2.text(-20, -28, f"({xpts_h:.2f})", fontproperties=font_props, color=color_home,
                ha='center', va='center', fontsize=xp_size, alpha=alpha, zorder=1)
    # Auswärts rechts
    ax2.text(26.25, 27, away_team, fontproperties=font_props, color=color_away,
                ha='center', va='center', fontsize=team_size, alpha=alpha, zorder=1)
    ax2.text(20, -3,   goals_a, fontproperties=font_props, color=color_away,
                ha='center', va='center', fontsize=goal_size, alpha=alpha, zorder=1)
    ax2.text(20, -18, f"{max_cum_xGAP_a_2:.2f}", fontproperties=font_props, color=color_away,
                ha='center', va='center', fontsize=xg_size, alpha=alpha, zorder=1)
    ax2.text(20, -28, f"({xpts_a:.2f})", fontproperties=font_props, color=color_away,
                ha='center', va='center', fontsize=xp_size, alpha=alpha, zorder=1)

    # -------------------------------------------------- xG-Timeline --------------------------------------------------
    # Achsen-Layout
    ax_h1.set_facecolor(background_color)
    ax_h1.spines[["top", "right"]].set_visible(False)
    ax_h1.spines[["bottom", "left"]].set_color("white")
    ax_h1.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color=text_color)
    ax_h2.set_facecolor(background_color)
    ax_h2.spines[["top", "left"]].set_visible(False)
    ax_h2.spines[["bottom", "right"]].set_color("white")
    ax_h2.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color=text_color)

    # y-Skala-Grenze bestimmen
    ymax = max(max_cum_xGAP_h_2, max_cum_xGAP_a_2)
    ax_h1.set_ylim(0, ymax + 0.5)
    ax_h1.yaxis.set_major_locator(MultipleLocator(0.5))

    # Stufendiagramm der 1. Halbzeit plotten
    ax_h1.step(Min_h_1, cum_xGAP_h_1, color=color_home, label=home_team, where="post", linewidth=2)
    ax_h1.step(Min_a_1, cum_xGAP_a_1, color=color_away, label=away_team, where="post", linewidth=2)
    ax_h1.set_title("1. Halbzeit", fontproperties=font_props, fontsize=15, color=text_color)
    ax_h1.set_xlim(0, end_1+1)
    ax_h1.set_xticks(list(range(0, end_1+1, 5)))
    ax_h1.set_ylabel("Kumulierte xGAP", fontproperties=font_props, fontsize=15, color=text_color)
    ax_h1.set_xlabel("Minute", fontproperties=font_props, fontsize=15, color=text_color)
    ax_h1.scatter(gMin_h_1, goals_h_1, s=80, marker="o", facecolors=color_home, edgecolors=background_color, zorder=5)
    ax_h1.scatter(gMin_a_1, goals_a_1, s=80, marker="o", facecolors=color_away, edgecolors=background_color, zorder=5)

    # Stufendiagramm der 2. Halbzeit plotten
    ax_h2.step(Min_h_2, cum_xGAP_h_2, color=color_home, label=home_team, where="post", linewidth=2)
    ax_h2.step(Min_a_2, cum_xGAP_a_2, color=color_away, label=away_team, where="post", linewidth=2)
    ax_h2.set_title("2. Halbzeit", fontproperties=font_props, fontsize=15, color=text_color)
    ax_h2.set_xlim(40, end_2+41)
    ax_h2.set_xticks(list(range(40, end_2+41, 5)))
    ax_h2.set_xlabel("Minute", fontproperties=font_props, fontsize=15, color=text_color)
    ax_h2.yaxis.tick_right()
    ax_h2.scatter(gMin_h_2, goals_h_2, s=80, marker="o", facecolors=color_home, edgecolors=background_color, zorder=5)
    ax_h2.scatter(gMin_a_2, goals_a_2, s=80, marker="o", facecolors=color_away, edgecolors=background_color, zorder=5)

    # Achsenbeschriftung anpassen
    for ticks in ax_h1.get_xticklabels() + ax_h1.get_yticklabels():
        ticks.set_fontproperties(font_props)
        ticks.set_fontsize(12)
        ticks.set_color(text_color)
    for ticks in ax_h2.get_xticklabels() + ax_h2.get_yticklabels():
        ticks.set_fontproperties(font_props)
        ticks.set_fontsize(12)
        ticks.set_color(text_color)
        
    # Y-Ticks auf eine Nachkommastelle runden
    ax_h1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_h2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    st.pyplot(fig)
    plt.close(fig)

# ================================================== SPIEL-ÜBERSICHT ==================================================
default_slider = (1, spiele["SID"].max())
default_heimauswärts = "Gesamt" 

# Session State initialisieren
if "start_end" not in st.session_state:
    st.session_state.start_end = default_slider
if "heimauswärts" not in st.session_state:
    st.session_state.heimauswärts = default_heimauswärts

def reset_filters_1():
    st.session_state.start_end = default_slider
    st.session_state.heimauswärts = default_heimauswärts

st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([77, 23])
with col1:
    st.subheader("Spiele")
with col2:
    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
    st.button("Filter zurücksetzen", on_click=reset_filters_1, key="reset_1")

col1, col2 = st.columns([59, 41])
with col1:
    start, end = st.slider("Spiele wählen", min_value=1, max_value=spiele["SID"].max(), key="start_end")
with col2:
    heimauswärts = st.radio("", ["Gesamt", "Heim", "Auswärts"], key="heimauswärts", horizontal=True)

spiele = spiele[spiele["SID"].between(start, end)].copy()
abschlüsse = abschlüsse[abschlüsse["SID"].between(start, end)].copy()
karten = karten[karten["SID"].between(start, end)].copy()
spielzeiten = spielzeiten[spielzeiten["SID"].between(start, end)].copy()
spiele_üb = spiele_üb[spiele_üb["SID"].between(start, end)].copy()

ergebnisse_fcn = spiele_üb[spiele_üb["Team"]=="FCN"].copy()
ergebnisse_opp = spiele_üb[spiele_üb["Team"]!="FCN"].copy()

abschlüsse_fcn = abschlüsse[abschlüsse["TID"]=="FCN"].copy()
abschlüsse_opp = abschlüsse[abschlüsse["TID"]!="FCN"].copy()

# -------------------------------------------------- Vorbereitungen xPLusMinus & Spielzeiten --------------------------------------------------
# Hilfsspalten erstellen
nummern = spieler["Nr"].unique()

@st.cache_data
def compute_player_on_pitch(df_abschlüsse, df_spielzeiten, nummern, hz_col_von, hz_col_bis):
    df_pop = df_abschlüsse.copy()
    for i in nummern:
        df_pop[f"Sp{i}"] = False  # Spalte anlegen

        # Nur weitermachen, wenn Spieler überhaupt Einsatzzeiten hat
        if i in df_spielzeiten["Nr"].values:
            zeiten = df_spielzeiten.loc[df_spielzeiten["Nr"] == i, ["SID", hz_col_von, hz_col_bis]].dropna()
            if not zeiten.empty:
                for _, row in zeiten.iterrows():
                    sid, von, bis = row["SID"], row[hz_col_von], row[hz_col_bis]
                    mask = (df_pop["SID"] == sid) & (df_pop["Min"].between(von, bis))
                    df_pop.loc[mask, f"Sp{i}"] = True

    return df_pop

# Für jede Halbzeit je eine Minutenliste erzeugen
spiele["min_HZ1"] = spiele.apply(lambda r: list(range(1, r["1. HZ"] + 1)), axis=1)
spiele["min_HZ2"] = spiele.apply(lambda r: list(range(41, 40 + r["2. HZ"] + 1)), axis=1)

# DataFrames für beide Halbzeiten bauen
df_hz1 = spiele[["SID", "min_HZ1"]].explode("min_HZ1")
df_hz1["HZ"] = 1
df_hz1 = df_hz1.rename(columns={"min_HZ1": "Min"})

df_hz2 = spiele[["SID", "min_HZ2"]].explode("min_HZ2")
df_hz2["HZ"] = 2
df_hz2 = df_hz2.rename(columns={"min_HZ2": "Min"})

# Beide zusammenfügen
df_minuten = pd.concat([df_hz1, df_hz2], ignore_index=True)
spiele.drop(["min_HZ1", "min_HZ2"], axis=1, inplace=True)

df_minuten["HG"] = df_minuten["SID"].isin(heimspiele).map({True: "H", False: "G"})

df_minuten_h1 = df_minuten[df_minuten["HZ"]==1].copy()
df_minuten_h2 = df_minuten[df_minuten["HZ"]==2].copy()

df_minuten_h1 = compute_player_on_pitch(df_minuten_h1, spielzeiten, nummern, "1Von", "1Bis")
df_minuten_h2 = compute_player_on_pitch(df_minuten_h2, spielzeiten, nummern, "2Von", "2Bis")

df_minuten = pd.concat([df_minuten_h1, df_minuten_h2], ignore_index=True)
df_minuten = df_minuten.sort_values(by=["SID", "HZ", "Min"], ascending=[True, True, True]).reset_index(drop=True)

tore_fcn = abschlüsse_fcn[abschlüsse_fcn["Ergeb"].isin(["Tor", "ET"])]
tore_opp = abschlüsse_opp[abschlüsse_opp["Ergeb"].isin(["Tor", "ET"])]

tore = pd.concat([tore_fcn, tore_opp], ignore_index=True)
tore = tore.sort_values(by=["SID", "HZ", "Min", "AP"], ascending=[True, True, True, True]).reset_index(drop=True)

@st.cache_data
def compute_gamestate(df_minuten, df_tore, team="FCN"):
    df_minuten = df_minuten.copy()
    # GS initialisieren
    df_minuten["GameState"] = 0

    # Tore sortieren nach SID, HZ und Min
    df_tore_sorted = df_tore.sort_values(["SID", "HZ", "Min"])

    # Schleife über jedes Spiel (SID) und jede Halbzeit (HZ)
    for sid in df_minuten["SID"].unique():
        gs = 0 # Startwert GS pro Spiel
        for hz in sorted(df_minuten["HZ"].unique()):
            df_game_hz = df_minuten[(df_minuten["SID"] == sid) & (df_minuten["HZ"] == hz)].sort_values("Min").reset_index()
            
            gs_list = []

            for idx, row in df_game_hz.iterrows():
                current_min = row["Min"]
                prev_min = current_min - 1

                # Tore in derselben SID, HZ und vorherigen Minute
                tore_prev = df_tore_sorted[
                    (df_tore_sorted["SID"] == sid) &
                    (df_tore_sorted["HZ"] == hz) &
                    (df_tore_sorted["Min"] == prev_min)
                ]

                for _, t_row in tore_prev.iterrows():
                    if t_row["TID"] == team:
                        gs += 1
                    else:
                        gs -= 1

                gs_list.append(gs)

            # Ergebnis ins df_minuten übernehmen
            mask = (df_minuten["SID"] == sid) & (df_minuten["HZ"] == hz)
            df_minuten.loc[mask, "GameState"] = gs_list
    
    return df_minuten

@st.cache_data
def compute_playerstate(df_minuten, df_karten, team="FCN"):
    df_minuten = df_minuten.copy()
    # PS initialisieren
    df_minuten["PlayerState"] = 0

    # Karten sortieren nach SID, HZ und Min
    df_karten_sorted = df_karten.sort_values(["SID", "HZ", "Min"])

    # Schleife über jedes Spiel (SID) und jede Halbzeit (HZ)
    for sid in df_minuten["SID"].unique():
        ps = 0 # Startwert GS pro Spiel
        for hz in sorted(df_minuten["HZ"].unique()):
            df_game_hz = df_minuten[(df_minuten["SID"] == sid) & (df_minuten["HZ"] == hz)].sort_values("Min").reset_index()
            
            ps_list = []

            for idx, row in df_game_hz.iterrows():
                current_min = row["Min"]

                # Karten in derselben SID, HZ und vorherigen Minute
                karten_min = df_karten_sorted[
                    (df_karten_sorted["SID"] == sid) &
                    (df_karten_sorted["HZ"] == hz) &
                    (df_karten_sorted["Min"] == current_min)
                ]

                for _, t_row in karten_min.iterrows():
                    if t_row["Team"] == team:
                        ps -= 1
                    else:
                        ps += 1

                ps_list.append(ps)

            # Ergebnis ins df_minuten übernehmen
            mask = (df_minuten["SID"] == sid) & (df_minuten["HZ"] == hz)
            df_minuten.loc[mask, "PlayerState"] = ps_list

    return df_minuten

df_minuten = compute_gamestate(df_minuten, tore, team="FCN")
df_minuten = compute_playerstate(df_minuten, karten, team="FCN")

df_minuten = df_minuten[df_minuten["SID"].between(start, end)].copy()

# -------------------------------------------------- Filter Heim / Auswärts --------------------------------------------------
if heimauswärts == "Heim":
    ergebnisse_fcn = ergebnisse_fcn[ergebnisse_fcn["HG"]=="H"]
    ergebnisse_opp = ergebnisse_opp[ergebnisse_opp["HG"]=="G"]
    abschlüsse_fcn = abschlüsse_fcn[abschlüsse_fcn["HG"]=="H"]
    abschlüsse_opp = abschlüsse_opp[abschlüsse_opp["HG"]=="G"]
    df_minuten = df_minuten[df_minuten["SID"].isin(heimspiele)]
elif heimauswärts == "Auswärts":
    ergebnisse_fcn = ergebnisse_fcn[ergebnisse_fcn["HG"]=="G"]
    ergebnisse_opp = ergebnisse_opp[ergebnisse_opp["HG"]=="H"]
    abschlüsse_fcn = abschlüsse_fcn[abschlüsse_fcn["HG"]=="G"]
    abschlüsse_opp = abschlüsse_opp[abschlüsse_opp["HG"]=="H"]
    df_minuten = df_minuten[df_minuten["SID"].isin(auswärtsspiele)]

if abschlüsse_fcn.empty and abschlüsse_opp.empty:
    st.error("Aktuell sind keine Abschlüsse ausgewählt!")
else:
    col1, col2, col3 = st.columns([39, 36, 25])

    with col1:
        st.markdown(f"Punkte: {int(ergebnisse_fcn["Punkte"].sum())} ({float(ergebnisse_fcn["xPoints"].sum().round(2)):.2f})")
    with col2:
        st.markdown(f"Tore: {int(ergebnisse_fcn["Tore"].sum())} ({float(ergebnisse_fcn["xG"].sum().round(2)):.2f})")
    with col3:
        st.markdown(f"Gegentore: {int(ergebnisse_opp["Tore"].sum())} ({float(ergebnisse_opp["xG"].sum().round(2)):.2f})")

# ================================================== SPIELER-METRIKEN ==================================================
# -------------------------------------------------- Filter Vorbereitungen --------------------------------------------------
default_modus = "Absolut"
options_gs = ["<-2", "-2", "-1", "0", "1", "2", ">2"]
options_ps = ["<-1", "-1", "0", "1", ">1"]
options_phase = ['Eigener Ballbesitz', 'Umschalten Offensiv', 'Ecke', 'Freistoss', 'Elfmeter']

# Positionen filtern
position_map = {
    "Tor": "TW",
    "Abwehr": "AB",
    "Mittelfeld": "MF",
    "Sturm": "ST"
}
options_position = list(position_map.keys())

# Session State initialisieren
if "modus" not in st.session_state:
    st.session_state.modus = default_modus
if "gs" not in st.session_state:
    st.session_state.gs = options_gs
if "ps" not in st.session_state:
    st.session_state.ps = options_ps
if "phase" not in st.session_state:
    st.session_state.phase = options_phase
if "position" not in st.session_state:
    st.session_state.position = options_position
if "min_spielzeit" not in st.session_state:
    st.session_state.min_spielzeit = 0

# Funktion zum Zurücksetzen
def reset_filters_2():
    st.session_state.modus = default_modus
    st.session_state.gs = options_gs
    st.session_state.ps = options_ps
    st.session_state.phase = options_phase
    st.session_state.position = options_position
    st.session_state.min_spielzeit = 0

# -------------------------------------------------- Filter erstellen  --------------------------------------------------
st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([77, 23])
with col1:
    st.subheader("Metriken")
with col2:
    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
    st.button("Filter zurücksetzen", on_click=reset_filters_2, key="reset_2")

col1, col2 = st.columns([2, 3])
with col1:
    modus = st.radio("Maßeinheit",["Absolut", "Pro 80 Minuten"], index=["Absolut", "Pro 80 Minuten"].index(st.session_state.modus), key="modus", horizontal=True)
with col2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    with st.expander("Weitere Filter"):
        gs = st.multiselect("GameState", options_gs, default=st.session_state.gs, key="gs")
        ps = st.multiselect("PlayerState", options_ps, default=st.session_state.ps, key="ps")

abschlüsse_fcn["gs_filt"] = abschlüsse_fcn["GS"].astype(str)
abschlüsse_fcn.loc[abschlüsse_fcn["GS"] > 2, "gs_filt"] = ">2"
abschlüsse_fcn.loc[abschlüsse_fcn["GS"] < -2, "gs_filt"] = "<-2"

df_minuten["gs_filt"] = df_minuten["GameState"].astype(str)
df_minuten.loc[df_minuten["GameState"] > 2, "gs_filt"] = ">2"
df_minuten.loc[df_minuten["GameState"] < -2, "gs_filt"] = "<-2"

abschlüsse_fcn["ps_filt"] = abschlüsse_fcn["PR"].astype(str)
abschlüsse_fcn.loc[abschlüsse_fcn["PR"] > 1, "ps_filt"] = ">1"
abschlüsse_fcn.loc[abschlüsse_fcn["PR"] < -1, "ps_filt"] = "<-1"

df_minuten["ps_filt"] = df_minuten["PlayerState"].astype(str)
df_minuten.loc[df_minuten["PlayerState"] > 1, "ps_filt"] = ">1"
df_minuten.loc[df_minuten["PlayerState"] < -1, "ps_filt"] = "<-1"

phase = st.multiselect("Spielphase", options_phase, default=st.session_state.phase, key="phase")

abschlüsse_fcn = abschlüsse_fcn[abschlüsse_fcn["gs_filt"].isin(gs)]
df_minuten = df_minuten[df_minuten["gs_filt"].isin(gs)]
abschlüsse_fcn = abschlüsse_fcn[abschlüsse_fcn["ps_filt"].isin(ps)]
df_minuten = df_minuten[df_minuten["ps_filt"].isin(ps)]
abschlüsse_fcn = abschlüsse_fcn[abschlüsse_fcn["Spielphase"].isin(phase)]

# Maximale Spielzeit berechnen
if heimauswärts == "Heim":
    spielzeit_max = (df_minuten["HG"]=="H").sum()
elif heimauswärts == "Auswärts":
    spielzeit_max = (df_minuten["HG"]=="G").sum()
else:
    spielzeit_max = len(df_minuten)

default_min_spielzeit = st.session_state.get("min_spielzeit", 0)
default_min_spielzeit = min(default_min_spielzeit, spielzeit_max)

col1, col2 = st.columns([2, 1])
with col1:
    position = st.multiselect("Position", options=options_position, default=st.session_state.position, key="position")
with col2:
    min_spielzeit = st.number_input("Mindestspielzeit", min_value=0, max_value=spielzeit_max, value=default_min_spielzeit, key="min_spielzeit")

gefilterte_positionen = [position_map[a] for a in position]

# -------------------------------------------------- Metriken berechnen --------------------------------------------------
startelf = (spielzeiten[spielzeiten["1Von"] == 1])["Nr"].value_counts().reset_index(name="Startelf")
spieler = spieler.merge(startelf, on="Nr", how="left")
spieler["Startelf"] = spieler["Startelf"].fillna(0).astype(int)

# Spielzeit pro Spieler berechnen
spielzeiten_dict = {i: df_minuten[f"Sp{i}"].sum() for i in nummern}
spielzeiten_es = pd.DataFrame(list(spielzeiten_dict.items()), columns=["Nr", "Spielzeit"])
spieler = spieler.merge(spielzeiten_es, on="Nr", how="left")

spieler["Spielzeitanteil"] = ((spieler["Spielzeit"]/spielzeit_max).round(2)*100).fillna(0).astype(int)

xg_pro_spieler = (abschlüsse_fcn.groupby("SNr")["xG"].sum().reset_index().rename(columns={"SNr": "Nr"}))
spieler = spieler.merge(xg_pro_spieler, on="Nr", how="left")
spieler["xG"] = spieler["xG"].fillna(0).round(2)

schüsse_pro_spieler = abschlüsse_fcn["SNr"].value_counts().reset_index()
schüsse_pro_spieler.columns = ["Nr", "Schüsse"]
spieler = spieler.merge(schüsse_pro_spieler, on="Nr", how="left")
spieler["Schüsse"] = (spieler["Schüsse"].fillna(0).round(2)).astype(int)

tore = abschlüsse_fcn[abschlüsse_fcn["Ergeb"]=="Tor"]

tore_pro_spieler = tore["SNr"].value_counts().reset_index()
tore_pro_spieler.columns = ["Nr", "Tore"]
spieler = spieler.merge(tore_pro_spieler, on="Nr", how="left")
spieler["Tore"] = (spieler["Tore"].fillna(0).round(2)).astype(int)

aufstor = abschlüsse_fcn[abschlüsse_fcn["Ergeb"].isin(["Tor", "Save", "TLK"])]

aufstor_pro_spieler = aufstor["SNr"].value_counts().reset_index()
aufstor_pro_spieler.columns = ["Nr", "Aufs Tor"]
spieler = spieler.merge(aufstor_pro_spieler, on="Nr", how="left")
spieler["Aufs Tor"] = (spieler["Aufs Tor"].fillna(0).round(2)).astype(int)

schlüsselpässe_pro_spieler = abschlüsse_fcn["VNr"].value_counts().reset_index()
schlüsselpässe_pro_spieler.columns = ["Nr", "Schlüsselpässe"]
spieler = spieler.merge(schlüsselpässe_pro_spieler, on="Nr", how="left")
spieler["Schlüsselpässe"] = (spieler["Schlüsselpässe"].fillna(0).round(2)).astype(int)

vorlagen_pro_spieler = tore["VNr"].value_counts().reset_index()
vorlagen_pro_spieler.columns = ["Nr", "Vorlagen"]
spieler = spieler.merge(vorlagen_pro_spieler, on="Nr", how="left")
spieler["Vorlagen"] = (spieler["Vorlagen"].fillna(0).round(2)).astype(int)

xa_pro_spieler = (abschlüsse_fcn.groupby("VNr")["xG"].sum().reset_index().rename(columns={"VNr": "Nr", "xG": "xA"}))
spieler = spieler.merge(xa_pro_spieler, on="Nr", how="left")
spieler["xA"] = spieler["xA"].fillna(0).round(2)

spieler["Effizienz"] = spieler["Tore"]-spieler["xG"]

# -------------------------------------------------- xGChain --------------------------------------------------
df_xgc = abschlüsse_fcn.copy()

invol = ['2VNr', '3VNr', '4VNr', '5VNr', '6VNr', '7VNr', '8VNr', '9VNr', '10VNr', '11VNr', '12VNr']

for i in invol:
    df_xgc.loc[df_xgc[i] == df_xgc["SNr"], i] = pd.NA
    df_xgc.loc[df_xgc[i] == df_xgc["VNr"], i] = pd.NA

invol.append("SNr")
invol.append("VNr")
dfs = []

for i in invol:
    temp = (df_xgc.groupby(i)["xG"].sum().reset_index().rename(columns={i: "Nr", "xG": "xGChain"}))
    dfs.append(temp)

xgc_pro_spieler = pd.concat(dfs, ignore_index=True)
xgc_pro_spieler = (xgc_pro_spieler.groupby("Nr", as_index=False)["xGChain"].sum())

spieler = spieler.merge(xgc_pro_spieler, on="Nr", how="left")
spieler["xGChain"] = spieler["xGChain"].fillna(0).round(2)

# -------------------------------------------------- xGBuildup --------------------------------------------------
df_xgb = abschlüsse_fcn.copy()

invol = ['2VNr', '3VNr', '4VNr', '5VNr', '6VNr', '7VNr', '8VNr', '9VNr', '10VNr', '11VNr', '12VNr']

dfs = []

for i in invol:
    temp = (df_xgb.groupby(i)["xG"].sum().reset_index().rename(columns={i: "Nr", "xG": "xGBuildup"}))
    dfs.append(temp)

xgb_pro_spieler = pd.concat(dfs, ignore_index=True)
xgb_pro_spieler = (xgb_pro_spieler.groupby("Nr", as_index=False)["xGBuildup"].sum())

spieler = spieler.merge(xgb_pro_spieler, on="Nr", how="left")
spieler["xGBuildup"] = spieler["xGBuildup"].fillna(0).round(2)

# -------------------------------------------------- xPlusMinus --------------------------------------------------
abschlüsse_fcn_1 = compute_player_on_pitch(abschlüsse_fcn[abschlüsse_fcn["HZ"] == 1], spielzeiten, nummern, "1Von", "1Bis")
abschlüsse_fcn_2 = compute_player_on_pitch(abschlüsse_fcn[abschlüsse_fcn["HZ"] == 2], spielzeiten, nummern, "2Von", "2Bis")
abschlüsse_opp_1 = compute_player_on_pitch(abschlüsse_opp[abschlüsse_opp["HZ"] == 1], spielzeiten, nummern, "1Von", "1Bis")
abschlüsse_opp_2 = compute_player_on_pitch(abschlüsse_opp[abschlüsse_opp["HZ"] == 2], spielzeiten, nummern, "2Von", "2Bis")

abschlüsse_fcn = pd.concat([abschlüsse_fcn_1, abschlüsse_fcn_2], ignore_index=True)
abschlüsse_opp = pd.concat([abschlüsse_opp_1, abschlüsse_opp_2], ignore_index=True)

abschlüsse_fcn = abschlüsse_fcn.sort_values(by=["SID", "HZ", "Min", "AP"], ascending=[True, True, True, True]).reset_index(drop=True)
abschlüsse_opp = abschlüsse_opp.sort_values(by=["SID", "HZ", "Min", "AP"], ascending=[True, True, True, True]).reset_index(drop=True)

import re

sp_cols = [c for c in abschlüsse_fcn.columns if re.fullmatch(r"Sp\d+", c)]

abschlüsse_fcn[sp_cols] = abschlüsse_fcn[sp_cols].fillna(False).astype(int)
abschlüsse_fcn[sp_cols] = abschlüsse_fcn[sp_cols].mul(abschlüsse_fcn["xGAP"], axis=0)

abschlüsse_opp[sp_cols] = abschlüsse_opp[sp_cols].fillna(False).astype(int)
abschlüsse_opp[sp_cols] = abschlüsse_opp[sp_cols].mul(abschlüsse_opp["xGAP"], axis=0)

xg_imp = (abschlüsse_fcn[sp_cols].sum(axis=0).reset_index().rename(columns={"index": "Spalte", 0: "xGImpact"}))
xg_imp["Nr"] = xg_imp["Spalte"].str.extract(r"(\d+)").astype(int)
xg_imp = xg_imp[["Nr", "xGImpact"]].sort_values("Nr").reset_index(drop=True)

xga_imp = (abschlüsse_opp[sp_cols].sum(axis=0).reset_index().rename(columns={"index": "Spalte", 0: "xGAImpact"}))
xga_imp["Nr"] = xga_imp["Spalte"].str.extract(r"(\d+)").astype(int)
xga_imp = xga_imp[["Nr", "xGAImpact"]].sort_values("Nr").reset_index(drop=True)

spieler = (spieler.merge(xg_imp, on="Nr", how="left").merge(xga_imp, on="Nr", how="left"))
spieler[["xGImpact", "xGAImpact"]] = spieler[["xGImpact", "xGAImpact"]].round(2)
spieler["xPlusMinus"] = spieler["xGImpact"]-spieler["xGAImpact"]

spieler = spieler[['Nr', 'Vorname', 'Nachname', 'Position', 'Startelf', 'Spielzeit', 'Spielzeitanteil', 
                   'Schüsse', 'Aufs Tor', 'Tore', 'xG', 'Effizienz', 'Schlüsselpässe', 'Vorlagen', 'xA',
                   'xGChain', 'xGBuildup', 'xGImpact', 'xGAImpact', 'xPlusMinus']]

# -------------------------------------------------- pro 80 Minuten-----------------------------------------
if abschlüsse_fcn.empty:
    st.error("Aktuell sind keine Abschlüsse ausgewählt!")
else:
    if modus == "Pro 80 Minuten":
        spieler["Schüsse"] = ((spieler["Schüsse"]/spieler["Spielzeit"])*80).round(2)
        spieler["Aufs Tor"] = ((spieler["Aufs Tor"]/spieler["Spielzeit"])*80).round(2)
        spieler["Tore"] = ((spieler["Tore"]/spieler["Spielzeit"])*80).round(2)
        spieler["xG"] = ((spieler["xG"]/spieler["Spielzeit"])*80).round(2)
        spieler["Effizienz"] = ((spieler["Effizienz"]/spieler["Spielzeit"])*80).round(2)
        spieler["Schlüsselpässe"] = ((spieler["Schlüsselpässe"]/spieler["Spielzeit"])*80).round(2)
        spieler["Vorlagen"] = ((spieler["Vorlagen"]/spieler["Spielzeit"])*80).round(2)
        spieler["xA"] = ((spieler["xA"]/spieler["Spielzeit"])*80).round(2)
        spieler["xGChain"] = ((spieler["xGChain"]/spieler["Spielzeit"])*80).round(2)
        spieler["xGBuildup"] = ((spieler["xGBuildup"]/spieler["Spielzeit"])*80).round(2)
        spieler["xGImpact"] = ((spieler["xGImpact"]/spieler["Spielzeit"])*80).round(2)
        spieler["xGAImpact"] = ((spieler["xGAImpact"]/spieler["Spielzeit"])*80).round(2)
        spieler["xPlusMinus"] = ((spieler["xPlusMinus"]/spieler["Spielzeit"])*80).round(2)

        spalten_2dp = ["Schüsse", "Aufs Tor", "Tore", "xG", "Effizienz", "Schlüsselpässe", "Vorlagen", "xA", "xGChain", "xGBuildup", "xGImpact", "xGAImpact", "xPlusMinus"]
    else:
        spalten_2dp = ["xG", "Effizienz", "xA", "xGChain", "xGBuildup", "xGImpact", "xGAImpact", "xPlusMinus"] 

    spieler_gefiltert = spieler[(spieler["Position"].isin(gefilterte_positionen)) & (spieler["Spielzeit"]>=min_spielzeit)]

    # --- Gesamt-Zeile erstellen ---
    #spalten = ["Schüsse", "Aufs Tor", "Tore", "xG", "Schlüsselpässe", "Vorlagen", "xA"]
    #sum_row = spieler_gefiltert[spalten].sum()
    #sum_row["Nr"] = 0
    #sum_row["Vorname"] = "Alle"
    #sum_row["Nachname"] = "Spieler"
    #sum_row["Position"] = "-"
    #sum_row["Startelf"] = spiele["SID"].max()
    #sum_row["Spielzeit"] = spiele["1. HZ"].sum()+spiele["2. HZ"].sum()
    #sum_row["Spielzeitanteil"] = 100
    #sum_row["Effizienz"] = sum_row["Tore"]-sum_row["xG"]

    #spieler_gefiltert = pd.concat([spieler_gefiltert, sum_row.to_frame().T], ignore_index=True)

    column_config = {
    "Nr": st.column_config.Column(pinned="left"),
    "Vorname": st.column_config.Column(pinned="left"),
    "Nachname": st.column_config.Column(pinned="left"),

    "Spielzeitanteil": st.column_config.NumberColumn(format="%.0f%%"),

    "Schüsse": st.column_config.NumberColumn("Schüsse", format="%.2f" if "Schüsse" in spalten_2dp else None),
    "Aufs Tor": st.column_config.NumberColumn("Aufs Tor", format="%.2f" if "Aufs Tor" in spalten_2dp else None),
    "Tore": st.column_config.NumberColumn("Tore", format="%.2f" if "Tore" in spalten_2dp else None),
    "xG": st.column_config.NumberColumn("xG", help="Summe der Torwahrscheinlichkeiten aller Schüsse", format="%.2f" if "xG" in spalten_2dp else None),
    "Effizienz": st.column_config.NumberColumn("Effizienz", help="Tore - xG", format="%.2f" if "Effizienz" in spalten_2dp else None),
    "Schlüsselpässe": st.column_config.NumberColumn("Schlüsselpässe", help="Pässe, die einen Schuss vorbereiten", format="%.2f" if "Schlüsselpässe" in spalten_2dp else None),
    "Vorlagen": st.column_config.NumberColumn("Vorlagen", format="%.2f" if "Vorlagen" in spalten_2dp else None),
    "xA": st.column_config.NumberColumn("xA", help="xG aller vorbereiteten Schüsse", format="%.2f" if "xA" in spalten_2dp else None),
    "xGChain": st.column_config.NumberColumn("xGChain", help="xG der Schüsse, bei denen der Spieler an der Passstafette beteiligt war", format="%.2f" if "xGChain" in spalten_2dp else None),
    "xGBuildup": st.column_config.NumberColumn("xGBuildup", help="xG der Schüsse, bei denen der Spieler an der Passstafette beteiligt war (ohne Schlüsselpässe und Schüsse)", format="%.2f" if "xGBuildup" in spalten_2dp else None),
    "xGImpact": st.column_config.NumberColumn("xGImpact", help="xG des eigenen Teams, während der Spieler auf dem Feld stand", format="%.2f" if "xGImpact" in spalten_2dp else None),
    "xGAImpact": st.column_config.NumberColumn("xGAImpact", help="xG des gegnerischen Teams, während der Spieler auf dem Feld stand", format="%.2f" if "xGAImpact" in spalten_2dp else None),
    "xPlusMinus": st.column_config.NumberColumn("xPlusMinus", help="xGImpact - xGAImpact", format="%.2f" if "xPlusMinus" in spalten_2dp else None),
    }

    st.dataframe(
        spieler_gefiltert,
        hide_index=True,
        column_config=column_config,
        use_container_width=True
    )

    # Spieler mit dem höchsten xG-Wert
    max_xG =int(spieler.loc[spieler["xG"] == spieler["xG"].max(), "Nr"].values[0])

# ================================================== SPIELER-MAP ==================================================
# Selectbox Spieler
optionen = spieler_filter["Name"].unique().tolist()
optionen.insert(0, "Alle Spieler")
#player_max_xG = spieler_filter.loc[spieler_filter["Nr"]==max_xG, "Name"].values[0]
#default_index = optionen.index(player_max_xG)

default_player = "Alle Spieler"
default_schussart = ['Links', 'Rechts', 'Kopf', 'Sonstige']
default_vorbereitung = ['Kontrollierte Vorlage', 'Hohe Flanke', 'Flache Hereingabe', 'Tiefer Pass', 'Dribbling', 'Direkter Standard', 'Unkontrollierte Vorbereitung']

# Session State initialisieren
if "player_filter" not in st.session_state:
    st.session_state.player_filter = default_player
if "schussart" not in st.session_state:
    st.session_state.schussart = default_schussart
if "vorbereitung" not in st.session_state:
    st.session_state.vorbereitung = default_vorbereitung

def reset_filters_3():
    st.session_state.player_filter = default_player
    st.session_state.schussart = default_schussart
    st.session_state.vorbereitung = default_vorbereitung 

st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([77, 23])
with col1:
    st.subheader("Einzelspieler")
with col2:
    st.markdown("<div style='height:0px'></div>", unsafe_allow_html=True)
    st.button("Filter zurücksetzen", on_click=reset_filters_3, key="reset_3")

col1, col2 = st.columns([42, 58])
with col1: 
    player_filter = st.selectbox("Spieler auswählen", options=optionen, key="player_filter")
    schussart = st.multiselect("Schussart", ['Links', 'Rechts', 'Kopf', 'Sonstige'], default=default_schussart, key="schussart")
with col2:
    vorbereitung = st.multiselect("Vorbereitungsart", ['Kontrollierte Vorlage', 'Hohe Flanke', 'Flache Hereingabe', 'Tiefer Pass', 'Dribbling', 'Direkter Standard', 'Unkontrollierte Vorbereitung'], 
        default=default_vorbereitung, key= "vorbereitung")

if player_filter == "Alle Spieler":
    einzelspieler = abschlüsse_fcn.copy()
    minuten = spielzeit_max
else:
    player = spieler_filter.loc[spieler_filter["Name"]==player_filter, "Nr"].values[0]
    einzelspieler = abschlüsse_fcn[abschlüsse_fcn["SNr"]==player].copy()
    minuten = int(spieler.loc[spieler["Nr"]==player, "Spielzeit"].values[0])

einzelspieler = einzelspieler[einzelspieler["Vorbereitung"].isin(vorbereitung)]
einzelspieler = einzelspieler[einzelspieler["Körperteil"].isin(schussart)]

if einzelspieler.empty:
    st.error("Aktuell sind keine Abschlüsse ausgewählt!")
else:
    einzelspieler_np = einzelspieler[einzelspieler["Spielphase"]!="Elfmeter"].copy()
    einzelspieler_p = einzelspieler[einzelspieler["Spielphase"]=="Elfmeter"].copy()

    # non-penalty
    xG_spieler = float(einzelspieler_np["xG"].sum().round(2))
    tore_spieler = int((einzelspieler_np["Ergeb"]=="Tor").sum())
    schüsse_spieler = int(einzelspieler_np["Ergeb"].notnull().sum())

    # penalty
    xG_spieler_p = float(einzelspieler_p["xG"].sum().round(2))
    tore_spieler_p = int((einzelspieler_p["Ergeb"]=="Tor").sum())
    schüsse_spieler_p = int(einzelspieler_p["Ergeb"].notnull().sum())

    if schüsse_spieler > 0:
        xg_pro_schuss = round((xG_spieler/schüsse_spieler), 2)
    else:
        xg_pro_schuss = "-"

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    fig.set_facecolor(background_color)

    gs = fig.add_gridspec(nrows = 6, ncols = 4)

    ax1 = fig.add_subplot(gs[0,0:4])
    ax2 = fig.add_subplot(gs[1:6,0:4])

    pitch = VerticalPitch(
        pitch_type='skillcorner', half=True, pitch_length=105, pitch_width=68,
        axis=False, label=False, tick=False,
        pad_left=3, pad_right=3, pad_top=3, pad_bottom=0.1,
        pitch_color=background_color, line_color=text_color,
        stripe=False, linewidth=1, corner_arcs=True, goal_type="box"
    )
    pitch.draw(ax=ax2)

    for i in einzelspieler.to_dict(orient="records"):
                pitch.scatter(
                    i["yFe"],
                    i["xFe"],
                    marker = '*' if i["Ergeb"] == "Tor" else 'o',
                    s = np.sqrt(i["xG"]) * 800 * (3 if i["Ergeb"] == "Tor" else 1),
                    facecolors=to_rgba("#AA1124", 0.5),
                    edgecolors=to_rgba("#AA1124", 1),
                    linewidth = 1.5,
                    zorder = 2,
                    ax = ax2)

    ax1.text(0.07, 0.7, player_filter, 
            fontproperties=font_props, color=text_color, ha='left', va='top', fontsize=30, alpha=1, zorder=1)
    ax1.text(0.07, 0.4, f"{minuten} (von {spielzeit_max}) Minuten gespielt", 
            fontproperties=font_props, color=text_color, ha='left', va='top', fontsize=20, alpha=1, zorder=1)
    ax1.text(0.07, 0.2, "U17 Bayernliga 2025/26", 
            fontproperties=font_props, color=text_color, ha='left', va='top', fontsize=20, alpha=1, zorder=1)
    ax1.text(0.93, 0.7, f"xG: {xG_spieler:.2f} ({tore_spieler} Tore / {schüsse_spieler} Schüsse)", 
            fontproperties=font_props, color=text_color, ha='right', va='top', fontsize=30, alpha=1, zorder=1)
    ax1.text(0.93, 0.4, f"+{xG_spieler_p:.2f} ({tore_spieler_p} Tore / {schüsse_spieler_p} Elfmeter)", 
            fontproperties=font_props, color=text_color, ha='right', va='top', fontsize=20, alpha=1, zorder=1)
    ax1.text(0.93, 0.2, f"xG/Schuss (ohne Elfmeter): {xg_pro_schuss:.2f}", 
            fontproperties=font_props, color=text_color, ha='right', va='top', fontsize=20, alpha=1, zorder=1)

    ax1.set_facecolor(background_color)
    ax1.axis("off")

    st.pyplot(fig)
    plt.close(fig)

import psutil

#process = psutil.Process()

#st.write(f"Aktueller RAM-Verbrauch: {process.memory_info().rss / 1024**2:.2f} MB")
