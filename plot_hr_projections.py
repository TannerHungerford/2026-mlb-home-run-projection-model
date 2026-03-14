import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("projected_2026_hr_dataset.csv")

top = df.head(15).copy()

# MLB team colors
team_colors = {
    "ATL":"#CE1141","ARI":"#A71930","BAL":"#DF4601","BOS":"#BD3039",
    "CHC":"#0E3386","CWS":"#27251F","CIN":"#C6011F","CLE":"#E31937",
    "COL":"#33006F","DET":"#0C2340","HOU":"#EB6E1F","KCR":"#004687",
    "LAA":"#BA0021","LAD":"#005A9C","MIA":"#00A3E0","MIL":"#12284B",
    "MIN":"#002B5C","NYM":"#002D72","NYY":"#003087","OAK":"#003831",
    "PHI":"#E81828","PIT":"#FDB827","SDP":"#2F241D","SEA":"#0C2C56",
    "SFG":"#FD5A1E","STL":"#C41E3A","TBR":"#092C5C","TEX":"#003278",
    "TOR":"#134A8E","WSN":"#AB0003"
}

# Create color list for bars
colors = [team_colors.get(team, "#888888") for team in top["Team"]]

plt.figure(figsize=(12,6))

plt.barh(
    top["Name"],
    top["Projected_HR"],
    color=colors
)

plt.xlabel("Projected Home Runs")
plt.title("Top 15 Projected MLB Home Run Hitters (2026)")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("hr_projection_chart.png", dpi=300)

plt.show()