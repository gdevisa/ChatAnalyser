import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from app import server
from app import app
# import all pages in the app
from apps import demo, instructions, home, results

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py

LOGO = "/assets/logo.png"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/home")),

    ],
    brand="Telegram Chat Analyzer",
    brand_href="/",
    color="primary",
    dark=True,
)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/demo_report':
        return demo.layout
    elif pathname == '/telegram_instructions':
        return instructions.layout
    elif pathname == '/results':
        return results.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)