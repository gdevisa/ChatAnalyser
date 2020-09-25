from app import app
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

text = dcc.Markdown('''
                1. Open Telegram Desktop on Windows.
                2. Open a desired chat (no group chat support at the moment).
                3. Click on menu button with three vertical dots.
                4. Select "Export chat history" from the menu.
                5. Leave ALL the boxes EMPTY, select Format: JSON, choose Path.
                6. Press Export.
''')

layout = html.Div(
    dbc.Container(
        dbc.Row(
            dbc.Card(
                dbc.CardBody(
                    [html.H4("Instructions", className="card-title"), text],
                    style={
                           "textAlign": "center",
                           "margin": "10px"}
                ),
            ), align='center', className='h-50', justify='center'
        ), style={"height": "100vh", }
    )
)

