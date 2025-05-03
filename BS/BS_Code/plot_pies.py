import json
import plotly.graph_objects as go
from pathlib import Path

# Load data
with open(Path(__file__).parent.parent / 'pie_plots' / 'utterances_test.json', encoding='utf-8') as f:
    data = json.load(f)

# Filter out 'TOTAL' entry
entries = [d for d in data if d['Intención'].lower() != 'total']

labels = [d['Intención'] for d in entries]
train_counts = [d['Número de utterances'] for d in entries]
test_counts = [d['Número de tests'] for d in entries]

# Assign consistent colors
import plotly.express as px
palette = px.colors.qualitative.Plotly
color_map = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
colors = [color_map[label] for label in labels]

# Pie for training
fig_train = go.Figure(go.Pie(labels=labels, values=train_counts, marker=dict(colors=colors),
                             textinfo='label+percent', hole=0.3, sort=False))
fig_train.update_layout(title_text='Distribución de ejemplos de entrenamiento por intención', font=dict(size=18))
fig_train.write_image(str(Path(__file__).parent.parent / 'pie_plots' / 'train_pie.pdf'), format='pdf', width=1200, height=900, scale=2)

# Pie for tests
fig_test = go.Figure(go.Pie(labels=labels, values=test_counts, marker=dict(colors=colors),
                            textinfo='label+percent', hole=0.3, sort=False))
fig_test.update_layout(title_text='Distribución de ejemplos de test por intención', font=dict(size=18))
fig_test.write_image(str(Path(__file__).parent.parent / 'pie_plots' / 'test_pie.pdf'), format='pdf', width=1200, height=900, scale=2)

print('Pie charts saved as train_pie.pdf and test_pie.pdf in pie_plots/')
