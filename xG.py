import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from mplsoccer import Pitch, VerticalPitch, Sbopen, FontManager

import football_functions_2 as ff

st.title("1. FC Nürnberg U16")
st.subheader("Expected Goals 2025/26")

# Neue Daten einlesen
df_new = pd.read_csv("xG/abschlüsse_xG.csv")
teams = pd.read_excel("xG/xG_U16_Anwendung.xlsx", sheet_name="Teams")
spiele = pd.read_excel("xG/xG_U16_Anwendung.xlsx", sheet_name="Spiele")
spieler = pd.read_excel("xG/xG_U16_Anwendung.xlsx", sheet_name="Spieler")
spielzeiten = pd.read_excel("xG/xG_U16_Anwendung.xlsx", sheet_name="Spielzeiten")
karten = pd.read_excel("xG/xG_U16_Anwendung.xlsx", sheet_name="Rote Karten")

# Grafik
game = st.selectbox("Spiel auswählen", spiele["SID"].sort_values().unique())

# Setting custom font
font_props = font_manager.FontProperties(fname="xG/dfb-sans-web-bold.64bb507.ttf")

teams["color"] = ["#AA1124", "#F8D615", "#CD1719", "#ED1248", "#006BB3", "#C20012", "#E3191B", "#03466A", 
                  "#2FA641", "#009C6B", "#ED1B24", "#E3000F", "#2E438C", "#5AAADF", "#EE232B"]
                  
from datetime import datetime
date = spiele.loc[spiele["SID"]==game, "Datum"].iloc[0]
date = date.strftime("%d.%m.%Y")

# Heim- und Auswärtsteam splitten
df_h = df_new[(df_new["SID"] == game) & (df_new["HG"] == "H")].copy()
df_a = df_new[(df_new["SID"] == game) & (df_new["HG"] == "G")].copy()
# Auswärtskoordianten spiegeln
df_a[["yFe", "xFe"]] *= -1

# Teamnamen und Farben
home_team_id = df_h.TID.unique()[0]
home_team = ff.find_club(teams, home_team_id)
color_home = ff.find_color(teams, home_team_id)

away_team_id = df_a.TID.unique()[0]
away_team = ff.find_club(teams, away_team_id)
color_away = ff.find_color(teams, away_team_id)

# Heim
# xGAP berechnen
xgap_h = df_h.groupby("AP")["xG"].apply(lambda x: 1 - np.prod(1 - x))

# max AbNr pro AP bestimmen
max_abnr_h = df_h.groupby("AP")["AbNr"].transform("max")

# neue Spalte xGAP setzen (nur bei höchster AbNr einer AP)
df_h["xGAP"] = np.where(
    df_h["AbNr"] == max_abnr_h,
    df_h["AP"].map(xgap_h),
    np.nan
)

# Gast
xgap_a = df_a.groupby("AP")["xG"].apply(lambda x: 1 - np.prod(1 - x))
max_abnr_a = df_a.groupby("AP")["AbNr"].transform("max")

df_a["xGAP"] = np.where(
    df_a["AbNr"] == max_abnr_a,
    df_a["AP"].map(xgap_a),
    np.nan
)

df_h["cum_xGAP"] = df_h["xGAP"].cumsum().where(df_h["xGAP"].notna())
df_a["cum_xGAP"] = df_a["xGAP"].cumsum().where(df_a["xGAP"].notna())

# Split 1. und 2. Halbzeit
df_h_1 = df_h[df_h["HZ"]==1].copy()
df_h_2 = df_h[df_h["HZ"]==2].copy()

df_a_1 = df_a[df_a["HZ"]==1].copy()
df_a_2 = df_a[df_a["HZ"]==2].copy()

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

df_goals = df_new[(df_new["SID"] == game) & df_new["Ergeb"].isin(["Tor", "ET"])].copy()

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
df_goals.loc[df_goals["Phase"]=="Elfmeter", "Entstehung"] = "11M"
df_goals.loc[df_goals["Ergeb"]=="ET", "Entstehung"] = "ET"

df_goals = df_goals[["Min", "Ereignis", "Spieler", "xG", "Entstehung", "Vorlage", "Vereinsname"]].copy()
df_goals["xG"] = df_goals["xG"].round(2)

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
        f"{'   ' + format(x['xG']*100, '.0f') + '%' if x['xG'] != 0 else ''}"  # xG-Wert
    ),
    axis=1
)

# --- xPoints ---
from scipy.stats import poisson

m_h_goals_probs = [poisson.pmf(i, max_cum_xGAP_h_2) for i in range(21)]
m_a_goals_probs = [poisson.pmf(i, max_cum_xGAP_a_2) for i in range(21)]

m_match_probs = np.outer(m_h_goals_probs, m_a_goals_probs)

m_p_h_win = np.sum(np.tril(m_match_probs, -1))
m_p_h_draw = np.sum(np.diag(m_match_probs))
m_p_h_loss = np.sum(np.triu(m_match_probs, 1))

m_xp_h = (m_p_h_win * 3) + (m_p_h_draw * 1) + (m_p_h_loss * 0)
m_xp_a = (m_p_h_win * 0) + (m_p_h_draw * 1) + (m_p_h_loss * 3)

# === AUSGABE ===
#GridSpec initialisieren
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# --- GridSpec-Layout ---
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

# --- Spielverlauf ---
ax1.set_facecolor(background_color)
ax1.axis("off")
ax1.text(0.5, 0.91, "Spielverlauf", ha="center", va="center", fontproperties=font_props, fontsize=30, color=text_color)
for i, txt in enumerate(liste):
    ax1.text(0.05, 0.85 - i * 0.05, txt, transform=ax1.transAxes, fontsize=14, fontproperties=font_props, va="top", ha="left", color=text_color)

# --- xG-Map ---
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

team_size = 25; goal_size = 150; xg_size = 50; alpha = 0.5; xp_size = 30
# Heim links
ax2.text(-26.25, 27, home_team, fontproperties=font_props, color=color_home,
            ha='center', va='center', fontsize=team_size, alpha=alpha, zorder=1)
ax2.text(-20, -3,   goals_h, fontproperties=font_props, color=color_home,
            ha='center', va='center', fontsize=goal_size, alpha=alpha, zorder=1)
ax2.text(-20, -18, f"{max_cum_xGAP_h_2:.2f}", fontproperties=font_props, color=color_home,
            ha='center', va='center', fontsize=xg_size, alpha=alpha, zorder=1)
ax2.text(-20, -28, f"({m_xp_h:.2f})", fontproperties=font_props, color=color_home,
            ha='center', va='center', fontsize=xp_size, alpha=alpha, zorder=1)
# Auswärts rechts
ax2.text(26.25, 27, away_team, fontproperties=font_props, color=color_away,
            ha='center', va='center', fontsize=team_size, alpha=alpha, zorder=1)
ax2.text(20, -3,   goals_a, fontproperties=font_props, color=color_away,
            ha='center', va='center', fontsize=goal_size, alpha=alpha, zorder=1)
ax2.text(20, -18, f"{max_cum_xGAP_a_2:.2f}", fontproperties=font_props, color=color_away,
            ha='center', va='center', fontsize=xg_size, alpha=alpha, zorder=1)
ax2.text(20, -28, f"({m_xp_a:.2f})", fontproperties=font_props, color=color_away,
            ha='center', va='center', fontsize=xp_size, alpha=alpha, zorder=1)

# --- xG-Timeline ---
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

# === METRIKEN ===
st.subheader("Metriken")

col1, col2 = st.columns(2)

with col1:
    start, end = st.slider("Spiele wählen", min_value=1, max_value=spiele["SID"].max(), value=[1, spiele["SID"].max()])
with col2:
    modus = st.radio("",["Absolut", "Pro 80 Minuten"], horizontal=True)

spiele = spiele[spiele["SID"].between(start, end)].copy()
abschlüsse = df_new[df_new["SID"].between(start, end)].copy()
spielzeiten = spielzeiten[spielzeiten["SID"].between(start, end)].copy()
karten = karten[karten["SID"].between(start, end)].copy()

startelf = (spielzeiten[spielzeiten["1Von"] == 1].copy())["Nr"].value_counts().reset_index(name="Startelf")
spieler = spieler.merge(startelf, on="Nr", how="left")
spieler["Startelf"] = spieler["Startelf"].fillna(0).astype(int)

spielzeit_max = spiele["1. HZ"].sum() + spiele["2. HZ"].sum()

spielzeit_sum = (spielzeiten.groupby("Nr", as_index=False).agg(Spielzeit=("SZ", "sum")))
spieler = spieler.merge(spielzeit_sum, on="Nr", how="left")
spieler["Spielzeit"] = spieler["Spielzeit"].fillna(0).astype(int)

spieler["Spielzeitanteil"] = ((spieler["Spielzeit"]/spielzeit_max).round(2)*100).astype(int)

xg_pro_spieler = (abschlüsse.groupby("SNr")["xG"].sum().reset_index().rename(columns={"SNr": "Nr"}))
spieler = spieler.merge(xg_pro_spieler, on="Nr", how="left")
spieler["xG"] = spieler["xG"].fillna(0).round(2)

schüsse_pro_spieler = abschlüsse["SNr"].value_counts().reset_index()
schüsse_pro_spieler.columns = ["Nr", "Schüsse"]
spieler = spieler.merge(schüsse_pro_spieler, on="Nr", how="left")
spieler["Schüsse"] = (spieler["Schüsse"].fillna(0).round(2)).astype(int)

tore = abschlüsse[abschlüsse["Ergeb"]=="Tor"].copy()

tore_pro_spieler = tore["SNr"].value_counts().reset_index()
tore_pro_spieler.columns = ["Nr", "Tore"]
spieler = spieler.merge(tore_pro_spieler, on="Nr", how="left")
spieler["Tore"] = (spieler["Tore"].fillna(0).round(2)).astype(int)

vorlagen_pro_spieler = tore["VNr"].value_counts().reset_index()
vorlagen_pro_spieler.columns = ["Nr", "Vorlagen"]
spieler = spieler.merge(vorlagen_pro_spieler, on="Nr", how="left")
spieler["Vorlagen"] = (spieler["Vorlagen"].fillna(0).round(2)).astype(int)

xa_pro_spieler = (abschlüsse.groupby("VNr")["xG"].sum().reset_index().rename(columns={"VNr": "Nr", "xG": "xA"}))
spieler = spieler.merge(xa_pro_spieler, on="Nr", how="left")
spieler["xA"] = spieler["xA"].fillna(0).round(2)

spieler["Effizienz"] = spieler["Tore"]-spieler["xG"]

# xGChain
df_xgc = abschlüsse.copy()

invol = ['VNr', '2VNr', '3VNr', '4VNr', '5VNr', '6VNr', '7VNr', '8VNr', '9VNr']

for i in invol:
    df_xgc.loc[df_xgc[i] == df_xgc["SNr"], i] = pd.NA

invol.append("SNr")
dfs = []

for i in invol:
    temp = (df_xgc.groupby(i)["xG"].sum().reset_index().rename(columns={i: "Nr", "xG": "xGChain"}))
    dfs.append(temp)

xgc_pro_spieler = pd.concat(dfs, ignore_index=True)
xgc_pro_spieler = (xgc_pro_spieler.groupby("Nr", as_index=False)["xGChain"].sum())

spieler = spieler.merge(xgc_pro_spieler, on="Nr", how="left")
spieler["xGChain"] = spieler["xGChain"].fillna(0).round(2)

# --- xPlusMinus ---
# xGAP berechnen
xgap = abschlüsse.groupby(["SID", "AP"])["xG"].apply(lambda x: 1 - np.prod(1 - x))

# max AbNr pro AP bestimmen
max_abnr = abschlüsse.groupby(["SID", "AP"])["AbNr"].transform("max")

# neue Spalte xGAP setzen (nur bei höchster AbNr einer AP)
abschlüsse["xGAP"] = np.where(
    abschlüsse["AbNr"] == max_abnr,
    abschlüsse.set_index(["SID", "AP"]).index.map(xgap),
    np.nan
)

abschlüsse_1 = abschlüsse[abschlüsse["HZ"] == 1].copy()
abschlüsse_2 = abschlüsse[abschlüsse["HZ"] == 2].copy()

# Hilfsspalten erstellen
nummern = spieler["Nr"].unique()

for i in nummern:
    abschlüsse_1[f"Sp{i}"] = False  # Spalte anlegen
    
    # Nur weitermachen, wenn Spieler überhaupt Einsatzzeiten in HZ1 hat
    if i in spielzeiten["Nr"].values:
        zeiten = spielzeiten.loc[spielzeiten["Nr"] == i, ["SID", "1Von", "1Bis"]].dropna()
        
        # Wenn es mindestens eine gültige Zeile gibt
        if not zeiten.empty:
            for _, row in zeiten.iterrows():
                sid = row["SID"]
                von = row["1Von"]
                bis = row["1Bis"]
                
                # Maske mit Pandas-Logik (das ist der eigentliche "if"-Inhalt)
                mask = (abschlüsse_1["SID"] == sid) & (abschlüsse_1["Min"].between(von, bis))
                abschlüsse_1.loc[mask, f"Sp{i}"] = True

for i in nummern:
    abschlüsse_2[f"Sp{i}"] = False  # Spalte anlegen
    
    # Nur weitermachen, wenn Spieler überhaupt Einsatzzeiten in HZ1 hat
    if i in spielzeiten["Nr"].values:
        zeiten = spielzeiten.loc[spielzeiten["Nr"] == i, ["SID", "2Von", "2Bis"]].dropna()
        
        # Wenn es mindestens eine gültige Zeile gibt
        if not zeiten.empty:
            for _, row in zeiten.iterrows():
                sid = row["SID"]
                von = row["2Von"]
                bis = row["2Bis"]
                
                # Maske mit Pandas-Logik (das ist der eigentliche "if"-Inhalt)
                mask = (abschlüsse_2["SID"] == sid) & (abschlüsse_2["Min"].between(von, bis))
                abschlüsse_2.loc[mask, f"Sp{i}"] = True

abschlüsse = pd.concat([abschlüsse_1, abschlüsse_2], ignore_index=True)

abschlüsse = abschlüsse.sort_values(by=["SID", "HZ", "Min", "AP"], ascending=[True, True, True, True]).reset_index(drop=True)

abschlüsse_for = abschlüsse[abschlüsse["TID"] == "FCN"].copy()
abschlüsse_aga = abschlüsse[abschlüsse["TID"] != "FCN"].copy()

import re

sp_cols = [c for c in abschlüsse_for.columns if re.fullmatch(r"Sp\d+", c)]

abschlüsse_for[sp_cols] = abschlüsse_for[sp_cols].fillna(False).astype(int)
abschlüsse_for[sp_cols] = abschlüsse_for[sp_cols].mul(abschlüsse_for["xGAP"], axis=0)

abschlüsse_aga[sp_cols] = abschlüsse_aga[sp_cols].fillna(False).astype(int)
abschlüsse_aga[sp_cols] = abschlüsse_aga[sp_cols].mul(abschlüsse_aga["xGAP"], axis=0)

xg_imp = (abschlüsse_for[sp_cols].sum(axis=0).reset_index().rename(columns={"index": "Spalte", 0: "xGImpact"}))
xg_imp["Nr"] = xg_imp["Spalte"].str.extract(r"(\d+)").astype(int)
xg_imp = xg_imp[["Nr", "xGImpact"]].sort_values("Nr").reset_index(drop=True)

xga_imp = (abschlüsse_aga[sp_cols].sum(axis=0).reset_index().rename(columns={"index": "Spalte", 0: "xGAImpact"}))
xga_imp["Nr"] = xga_imp["Spalte"].str.extract(r"(\d+)").astype(int)
xga_imp = xga_imp[["Nr", "xGAImpact"]].sort_values("Nr").reset_index(drop=True)

spieler = (spieler.merge(xg_imp, on="Nr", how="left").merge(xga_imp, on="Nr", how="left"))
spieler[["xGImpact", "xGAImpact"]] = spieler[["xGImpact", "xGAImpact"]].round(2)
spieler["xPlusMinus"] = spieler["xGImpact"]-spieler["xGAImpact"]

spieler = spieler[['Nr', 'Vorname', 'Nachname', 'Startelf', 'Spielzeit', 'Spielzeitanteil', 
                   'Schüsse', 'Tore', 'xG', 'Effizienz', 'Vorlagen', 'xA',
                   'xGChain', 'xGImpact', 'xGAImpact', 'xPlusMinus']]

if modus == "Pro 80 Minuten":
    spieler["Schüsse"] = ((spieler["Schüsse"]/spieler["Spielzeit"])*80).round(2)
    spieler["Tore"] = ((spieler["Tore"]/spieler["Spielzeit"])*80).round(2)
    spieler["Tore"] = ((spieler["Tore"]/spieler["Spielzeit"])*80).round(2)
    spieler["Effizienz"] = ((spieler["Effizienz"]/spieler["Spielzeit"])*80).round(2)
    spieler["Vorlagen"] = ((spieler["Vorlagen"]/spieler["Spielzeit"])*80).round(2)
    spieler["xA"] = ((spieler["xA"]/spieler["Spielzeit"])*80).round(2)
    spieler["xGChain"] = ((spieler["xGChain"]/spieler["Spielzeit"])*80).round(2)
    spieler["xGImpact"] = ((spieler["xGImpact"]/spieler["Spielzeit"])*80).round(2)
    spieler["xGAImpact"] = ((spieler["xGAImpact"]/spieler["Spielzeit"])*80).round(2)
    spieler["xPlusMinus"] = ((spieler["xPlusMinus"]/spieler["Spielzeit"])*80).round(2)

st.dataframe(spieler, hide_index=True)