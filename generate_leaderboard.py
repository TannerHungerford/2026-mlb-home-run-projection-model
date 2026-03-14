import pandas as pd

df = pd.read_csv("projected_2026_hr_dataset.csv")

top50 = df.head(50).copy()
top50["Projected_HR"] = top50["Projected_HR"].round(1)
top50.insert(0, "Rank", range(1, len(top50)+1))

html_table = top50[["Rank","Name","Team","Projected_HR"]].to_html(index=False)

html_page = f"""
<html>
<head>
<title>2026 MLB HR Projections</title>

<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
}}

h1 {{
    text-align: center;
}}

.table-container {{
    height: 600px;
    overflow-y: scroll;
    border: 1px solid #ccc;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    position: sticky;
    top: 0;
    background: #e6e6e6;
    font-weight: bold;
    padding: 10px;
}}

td {{
    padding: 8px;
    border-bottom: 1px solid #ddd;
}}

tr:nth-child(even) {{
    background-color: #f7f7f7;
}}

</style>

</head>

<body>

<h1>Top 50 Projected MLB Home Run Hitters (2026)</h1>

<div class="table-container">
{html_table}
</div>

</body>
</html>
"""

with open("hr_leaderboard.html", "w") as f:
    f.write(html_page)

print("Scrollable leaderboard created: hr_leaderboard.html")