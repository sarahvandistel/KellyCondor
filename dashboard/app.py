"""
Dash dashboard for KellyCondor performance visualization.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Initialize the Dash app
app = dash.Dash(__name__, title="KellyCondor Dashboard")

# Sample data for demonstration
def generate_sample_data():
    """Generate sample data for the dashboard."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Simulate equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity_curve = (1 + pd.Series(returns)).cumprod() * 100000
    
    # Simulate trade data
    trades = []
    for i in range(50):
        trade_date = np.random.choice(dates)
        pnl = np.random.normal(500, 2000)
        trades.append({
            'date': trade_date,
            'pnl': pnl,
            'type': 'iron_condor',
            'size': abs(pnl) / 1000
        })
    
    return equity_curve, pd.DataFrame(trades)


equity_curve, trades_df = generate_sample_data()

# Layout
app.layout = html.Div([
    html.H1("KellyCondor Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Summary metrics
    html.Div([
        html.Div([
            html.H3("Total Return", style={'textAlign': 'center'}),
            html.H2(f"${equity_curve.iloc[-1] - equity_curve.iloc[0]:,.0f}", 
                   style={'textAlign': 'center', 'color': '#27ae60'})
        ], className='four columns'),
        
        html.Div([
            html.H3("Win Rate", style={'textAlign': 'center'}),
            html.H2(f"{(trades_df['pnl'] > 0).mean():.1%}", 
                   style={'textAlign': 'center', 'color': '#3498db'})
        ], className='four columns'),
        
        html.Div([
            html.H3("Sharpe Ratio", style={'textAlign': 'center'}),
            html.H2(f"{trades_df['pnl'].mean() / trades_df['pnl'].std():.2f}", 
                   style={'textAlign': 'center', 'color': '#e74c3c'})
        ], className='four columns'),
    ], className='row', style={'marginBottom': 30}),
    
    # Charts
    html.Div([
        # Equity curve
        html.Div([
            html.H3("Equity Curve", style={'textAlign': 'center'}),
            dcc.Graph(
                id='equity-chart',
                figure={
                    'data': [
                        go.Scatter(
                            x=equity_curve.index,
                            y=equity_curve.values,
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#2ecc71', width=2)
                        )
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Portfolio Value ($)'},
                        hovermode='x unified'
                    )
                }
            )
        ], className='six columns'),
        
        # Trade distribution
        html.Div([
            html.H3("Trade PnL Distribution", style={'textAlign': 'center'}),
            dcc.Graph(
                id='pnl-dist-chart',
                figure=px.histogram(
                    trades_df, 
                    x='pnl', 
                    nbins=20,
                    title='Trade PnL Distribution',
                    color_discrete_sequence=['#3498db']
                )
            )
        ], className='six columns'),
    ], className='row'),
    
    # Kelly sizing metrics
    html.Div([
        html.H3("Kelly Sizing Metrics", style={'textAlign': 'center', 'marginTop': 30}),
        html.Div([
            html.Div([
                html.H4("Current IV Rank"),
                html.H2("0.65", style={'color': '#e67e22'})
            ], className='three columns'),
            
            html.Div([
                html.H4("Volatility Skew"),
                html.H2("0.12", style={'color': '#9b59b6'})
            ], className='three columns'),
            
            html.Div([
                html.H4("Kelly Fraction"),
                html.H2("0.18", style={'color': '#f39c12'})
            ], className='three columns'),
            
            html.Div([
                html.H4("Position Size"),
                html.H2("$1,200", style={'color': '#1abc9c'})
            ], className='three columns'),
        ], className='row')
    ]),
    
    # Recent trades table
    html.Div([
        html.H3("Recent Trades", style={'textAlign': 'center', 'marginTop': 30}),
        html.Div([
            html.Table(
                id='trades-table',
                children=[
                    html.Thead([
                        html.Tr([
                            html.Th("Date"),
                            html.Th("Type"),
                            html.Th("PnL"),
                            html.Th("Size")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(trade['date'].strftime('%Y-%m-%d')),
                            html.Td(trade['type']),
                            html.Td(f"${trade['pnl']:,.0f}"),
                            html.Td(f"{trade['size']:.1f}")
                        ]) for trade in trades_df.head(10).to_dict('records')
                    ])
                ],
                style={'width': '100%', 'textAlign': 'center'}
            )
        ])
    ])
], style={'padding': '20px'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 