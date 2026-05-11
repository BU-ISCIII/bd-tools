import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import requests


app = dash.Dash(__name__)


data = {
    "province": ["Madrid", "Barcelona", "Sevilla", "A Coruña"],
    "value": [100, 200, 150, 180]
}
df = pd.DataFrame(data)


geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-provinces.geojson"
geojson_data = requests.get(geojson_url).json()


geojson_names = [feature["properties"]["name"] for feature in geojson_data["features"]]


fig = px.choropleth_mapbox(
    df,
    geojson=geojson_data,
    locations="province",  # Columna del DataFrame
    featureidkey="properties.name",  # Llave del GeoJSON
    color="value",  # Columna para colorear
    color_continuous_scale="Blues",  # Escala de colores
    mapbox_style="carto-positron",  # Estilo del mapa base
    zoom=5,  # Nivel de zoom inicial
    center={"lat": 40.4168, "lon": -3.7038},  # Centro del mapa
    title=""
)

fig.update_layout(
    margin={"r": 20, "t": 20, "l": 20, "b": 20},  
    height=800, 
    showlegend=False 
)
fig.update_traces(
    hoverinfo="none",  # Deshabilitar el hover
    selector=dict(type="choroplethmapbox")
)

app.layout = html.Div([
    dcc.Graph(
        figure=fig,
        config={
            "displayModeBar": False 
        }
    )
])

if __name__ == "__main__":
    app.run_server(port=8051, debug=True)
