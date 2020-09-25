import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

layout = html.Div([
    dbc.Container([
        dbc.Container(
            dbc.Row(
                dbc.Card(
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem("Demo Report", href='/demo_report'),
                            dbc.ListGroupItem("Telegram Instructions", href='/telegram_instructions'),
                            dbc.ListGroupItem(
                                dcc.Upload(
                                    id='upload-file',
                                    children=html.Div([
                                        dbc.Button("Upload File (JSON)", color="primary", outline=True, className="mr-1"),
                                    ]),
                                    style={
                                        'textAlign': 'center',
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=False
                                ),
                            ),
                            dbc.ListGroupItem(
                                dbc.Button("Go", color="primary", className="mr-1", href='/results')
                            )
                        ],
                        flush=True,
                    ),
                    style={"width": "50%",
                           "textAlign": "center"},
                ), align= 'center', className='h-50', justify='center'
            ), style={"height": "100vh",}
        ),
        
       html.Div(id='output-file-upload')
    ])
])
