"""Prediction chart with explicit explanations for every unevaluated new day."""
from html import escape

import pandas as pd
import plotly.graph_objects as go

from utils.offshore_final_data import TZ, grid


def short_gap_label(reason):
    """Keep every known cause visible; leave technical details in the hover text."""
    rules = [
        ('No ECMWF run', 'No ECMWF run'),
        ('Incomplete ECMWF', 'Incomplete ECMWF data'),
        ('No complete ECMWF', 'Incomplete ECMWF data'),
        ('Weather request failed', 'Weather request failed'),
        ('Actual generation', 'Missing generation'),
        ('Installed capacity', 'Missing/stale capacity'),
        ('Capacity factor', 'Invalid capacity factor'),
        ('Training history', 'Insufficient training history'),
        ('Not evaluated', 'Evaluation not run'),
    ]
    labels = [label for match, label in rules if match.lower() in reason.lower()]
    # Do not silently hide a new, unrecognised diagnostic.
    return '\n'.join(dict.fromkeys(labels)) if labels else reason


def prediction_chart(predictions, dataset, coverage, evaluation_audit, selection):
    """Use the requested timeline, including missing days at either end."""
    first = pd.to_datetime(predictions.target_time, utc=True).min()
    last = pd.to_datetime(predictions.target_time, utc=True).max()
    if not coverage.empty:
        first = min(first, grid(coverage.index.min())[0])
        last = max(last, grid(coverage.index.max())[-1])
    timeline = pd.date_range(first, last, freq='15min')
    figure = go.Figure()
    # Actual values can exist even when weather is missing. Keep them visible.
    actual = dataset.reindex(timeline)
    usable_actual = actual.generation_available.fillna(False) & actual.capacity_available.fillna(False) & actual.target_valid.fillna(False)
    figure.add_scatter(x=timeline, y=actual.capacity_factor.where(usable_actual),
        name='Actual CF', mode='lines', line=dict(color='black', width=1.5), connectgaps=False)
    for name, frame in predictions.groupby('model'):
        series = frame.set_index(pd.to_datetime(frame.target_time, utc=True)).prediction
        figure.add_scatter(x=timeline, y=series.reindex(timeline),
            name=f"{name} ({selection['best_by_model'][name]['loss']})", mode='lines',
            line=dict(width=1), opacity=0.8, connectgaps=False)
    evaluated = set()
    skipped = {}
    if not evaluation_audit.empty:
        for day, rows in evaluation_audit.groupby('day'):
            if rows.status.eq('evaluated').all():
                evaluated.add(pd.Timestamp(day))
            else:
                skipped[pd.Timestamp(day)] = '; '.join(rows.reason.dropna().unique())
    gaps = []
    for day, row in coverage.iterrows():
        if day in evaluated:
            continue
        reason = row.get('reason', '') or skipped.get(day, '') or 'Not evaluated in this run (disabled or day limit)'
        gaps.append(dict(day=day, reason=reason))
    # Merge adjacent days with the same explanation to keep the plot readable.
    spans = []
    for gap in gaps:
        day, reason = gap['day'], gap['reason']
        end = day + pd.DateOffset(days=1)
        if spans and spans[-1]['end'] == day and spans[-1]['reason'] == reason:
            spans[-1]['end'] = end
        else:
            spans.append(dict(start=day, end=end, reason=reason, label=short_gap_label(reason)))
    for span in spans:
        start = span['start'].tz_localize(TZ).tz_convert('UTC')
        end = span['end'].tz_localize(TZ).tz_convert('UTC')
        middle = start + (end - start) / 2
        reason = escape(span['reason'])
        figure.add_shape(type='rect', x0=start.isoformat(), x1=end.isoformat(), y0=0, y1=1,
            xref='x', yref='paper', fillcolor='gray', opacity=0.22, line_width=0, layer='below')
        figure.add_annotation(x=middle.isoformat(), y=0.98, xref='x', yref='paper',
            text=escape(span['label']).replace('\n', '<br>'),
            hovertext=reason, showarrow=False, textangle=-90,
            font=dict(size=10, color='#555'))
        # A marker makes the full reason discoverable in hover mode too.
        figure.add_scatter(x=[middle], y=[0.95], mode='markers',
            marker=dict(color='gray', size=9, symbol='x'), showlegend=False,
            text=[reason], hovertemplate='%{text}<extra>Missing forecast</extra>')
    for date, label in [('2026-05-15', 'First holdout'), ('2026-06-12', 'Latest holdout')]:
        boundary = pd.Timestamp(date, tz=TZ).tz_convert('UTC').isoformat()
        figure.add_shape(type='line', x0=boundary, x1=boundary, y0=0, y1=1,
            xref='x', yref='paper', line=dict(color='red', dash='dash'))
        figure.add_annotation(x=boundary, y=1, xref='x', yref='paper', text=label, showarrow=False)
    figure.update_layout(title='Frozen offshore models: development and both holdouts',
        template='plotly_white', height=600, hovermode='x unified',
        xaxis=dict(title='Target time (UTC)', rangeslider=dict(visible=True)),
        yaxis=dict(title='Capacity factor', range=[-0.02, 1.02]))
    return figure, pd.DataFrame(spans, columns=['start', 'end', 'reason', 'label'])
