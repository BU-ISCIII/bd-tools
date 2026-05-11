import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, dcc
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

output_dir = "/home/sergio/git/dev/mepram_testing/bd-tools/web_app/output"
best_xgb_model = joblib.load(f"{output_dir}/xgboost_model.pkl")
scaler = joblib.load(f"{output_dir}/scaler.pkl")



app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.LUX,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
])


# Layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Sepsis Prediction Model", className="title"), width=12)
    ], className="mb-4"),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Video(
                        src="/assets/background2.mp4",
                        autoPlay=True,
                        loop=True,
                        muted=True,
                        className="card-video-background",
                        style={"position": "absolute", "top": 0, "left": 0, "width": "100%", "height": "100%", "zIndex": -1, "objectFit": "cover"}
                    ),
                    dbc.CardHeader("Parámetros del Paciente", className="text-center text-white bg-dark"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Accidente Cerebrovascular"),
                                    dbc.Select(id="e_cerebrovascular", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Linfoma"),
                                    dbc.Select(id="linfoma", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 2}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Respiración Score SOFA"),
                                    dbc.Input(id="respiracion", type="number", value=0, min=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Cardiovascular Score SOFA"),
                                    dbc.Input(id="cardiovascular", type="number", value=0, min=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Hipotensión"),
                                    dbc.Select(id="hipotension", options=[
                                        {"label": "No (>100 mm[Hg])", "value": 0},
                                        {"label": "Sí (<=100 mm[Hg])", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Síntoma Tos (días)"),
                                    dbc.Input(id="sintoma_2_0", type="number", value=1.0, min=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Dolor en el ángulo renal (días)"),
                                    dbc.Input(id="sintoma_8_0", type="number", value=1.0, min=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Creatinina Renal Score SOFA"),
                                    dbc.Input(id="creatinina", type="number", value=1.0, min=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Frecuencia Cardíaca (lpm)"),
                                    dbc.Input(id="frec_cardiaca", type="number", value=75, min=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Índice de Charlson"),
                                    dbc.Input(id="indice_de_charlson", type="number", value=0, min=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Naúseas (días)"),
                                    dbc.Input(id="sintoma_9_0", type="number", value=0, min=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Uso de Vasopresores"),
                                    dbc.Select(id="vasopresores", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Estado Mental Alterado"),
                                    dbc.Select(id="estado_mental_alterado", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Hipoxemia"),
                                    dbc.Select(id="hipoxemia", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Catéter Venoso"),
                                    dbc.Select(id="cateter_venoso", options=[
                                        {"label": "No", "value": 0},
                                        {"label": "Sí", "value": 1}
                                    ], value=0, className="mb-3"),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Tiempo desde última infección (días)"),
                                    dbc.Input(id="tiempo_ultima", type="number", value=0, min=0, className="mb-3"),
                                ], width=6),
                            ]),
                            html.Button([
                                html.Span(className="circle", children=[
                                    html.Span(className="icon arrow")
                                ]),
                                html.Span("Predict", className="button-text")
                            ], id="predict-button", className="predict-more")
                        ])
                ], className="custom-card mb-3")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    html.Video(
                        src="/assets/background2.mp4",
                        autoPlay=True,
                        loop=True,
                        muted=True,
                        className="card-video-background",
                        style={"position": "absolute", "top": 0, "left": 0, "width": "100%", "height": "100%", "zIndex": -1, "objectFit": "cover"}
                    ),
                    dbc.CardHeader("Resultados de Predicción", className="text-center text-white bg-dark"),
                    dbc.CardBody([
                        html.Div(id="prediction-output", className="text-center mb-4"),
                        html.Div(id="person-icon", className="text-center", children=[
                            html.I(className="fa-solid fa-person")
                        ])
                    ])
                ], className="custom-card mb-3"),

                dbc.Card([
                    dbc.CardHeader("Gráfico PCA", className="text-center text-white bg-dark"),
                    dbc.CardBody([
                        dcc.Graph(id="pca-graph",config={"responsive": True}, style={"width": "90%", "height": "60vh"})
                    ])
                ], className="custom-card mb-3", style={"background": "#ffffff"})
            ], width=8),
        ])
    ], fluid=True, className="content")
])


# Función de Predicción
def make_prediction(e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
                    sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
                    sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
                    cateter_venoso, tiempo_ultima):

    new_patient = pd.DataFrame([[ 
        e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
        sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
        sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
        cateter_venoso, tiempo_ultima
    ]], columns=[
        "e_cerebrovascular", "linfoma", "respiracion", "cardiovascular", "hipotension",
        "sintoma_2_0", "sintoma_8_0", "creatinina", "frec_cardiaca", "indice_de_charlson",
        "sintoma_9_0", "vasopresores", "estado_mental_alterado", "hipoxemia", 
        "cateter_venoso", "tiempo_ultima"
    ])

    column_mapping = {
        "sintoma_2_0": "sintoma_2.0",
        "sintoma_8_0": "sintoma_8.0",
        "sintoma_9_0": "sintoma_9.0"
    }
    new_patient.rename(columns=column_mapping, inplace=True)

    new_patient_scaled = scaler.transform(new_patient)

    avg_prob = (
        best_xgb_model.predict_proba(new_patient_scaled)[0, 1])

    if avg_prob < 0.3:
        color = "green"
    elif avg_prob < 0.5:
        color = "yellow"
    elif avg_prob < 0.7:
        color = "orange"
    else:
        color = "red"

    prediction_text = f"Probabilidad promedio: {avg_prob:.2f}"
    icon_style = {"color": color, "font-size": "150px"}

    return prediction_text, icon_style

# Función para el Gráfico PCA
def create_pca_graph(new_patient_data=None):
    cohort_data = pd.read_csv("/home/sergio/git/dev/mepram_testing/bd-tools/web_app/output/processed_df.csv")
    features = cohort_data.drop(columns=["sepsis"])
    sepsis_labels = cohort_data["sepsis"]

    pca_scaler = StandardScaler()
    cohort_scaled = pca_scaler.fit_transform(features)

    pca = PCA(n_components=3)
    cohort_pca = pca.fit_transform(cohort_scaled)

    cohort_df = pd.DataFrame(cohort_pca, columns=["PC1", "PC2", "PC3"])
    cohort_df["type"] = sepsis_labels.map({0: "Negativo", 1: "Positivo"})

    if new_patient_data is not None:
        new_patient_scaled = pca_scaler.transform(new_patient_data)
        new_patient_pca = pca.transform(new_patient_scaled)
        new_patient_df = pd.DataFrame(new_patient_pca, columns=["PC1", "PC2", "PC3"])
        new_patient_df["type"] = "Nuevo Paciente"
        cohort_df = pd.concat([cohort_df, new_patient_df], ignore_index=True)

    fig = px.scatter_3d(
        cohort_df, x="PC1", y="PC2", z="PC3", color="type",
        color_discrete_map={"Negativo": "#095171", "Positivo": "#79ddff", "Nuevo Paciente": "red"},
        title="Análisis de Componentes Principales (PCA 3D)"
    )
    fig.update_layout(
        legend_title_text="Sepsis",
        legend=dict(x=0.8, y=1),
        template= "plotly_white",
        scene=dict(
        aspectmode="cube"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))

    return fig

@app.callback(
    [Output("prediction-output", "children"),
     Output("person-icon", "style")],
    Input("predict-button", "n_clicks"),
    [State("e_cerebrovascular", "value"),
     State("linfoma", "value"),
     State("respiracion", "value"),
     State("cardiovascular", "value"),
     State("hipotension", "value"),
     State("sintoma_2_0", "value"), 
     State("sintoma_8_0", "value"), 
     State("creatinina", "value"),
     State("frec_cardiaca", "value"),
     State("indice_de_charlson", "value"),
     State("sintoma_9_0", "value"),
     State("vasopresores", "value"),
     State("estado_mental_alterado", "value"),
     State("hipoxemia", "value"),
     State("cateter_venoso", "value"),
     State("tiempo_ultima", "value")]
)
def handle_prediction(n_clicks, e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
                      sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
                      sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
                      cateter_venoso, tiempo_ultima):
    if n_clicks is None:
        return "Introduce los valores y presiona 'Predecir'.", {"color": "white", "font-size": "150px"}

    return make_prediction(e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
                           sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
                           sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
                           cateter_venoso, tiempo_ultima)


@app.callback(
    Output("pca-graph", "figure"),
    Input("predict-button", "n_clicks"),
    [State("e_cerebrovascular", "value"),
     State("linfoma", "value"),
     State("respiracion", "value"),
     State("cardiovascular", "value"),
     State("hipotension", "value"),
     State("sintoma_2_0", "value"),  
     State("sintoma_8_0", "value"),  
     State("creatinina", "value"),
     State("frec_cardiaca", "value"),
     State("indice_de_charlson", "value"),
     State("sintoma_9_0", "value"),
     State("vasopresores", "value"),
     State("estado_mental_alterado", "value"),
     State("hipoxemia", "value"),
     State("cateter_venoso", "value"),
     State("tiempo_ultima", "value")]
)
def handle_pca_graph(n_clicks, e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
                     sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
                     sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
                     cateter_venoso, tiempo_ultima):
    if n_clicks is None:
        return create_pca_graph()

    new_patient_data = pd.DataFrame([[ 
        e_cerebrovascular, linfoma, respiracion, cardiovascular, hipotension,
        sintoma_2_0, sintoma_8_0, creatinina, frec_cardiaca, indice_de_charlson,
        sintoma_9_0, vasopresores, estado_mental_alterado, hipoxemia, 
        cateter_venoso, tiempo_ultima
    ]], columns=[
        "e_cerebrovascular", "linfoma", "respiracion", "cardiovascular", "hipotension",
        "sintoma_2_0", "sintoma_8_0", "creatinina", "frec_cardiaca", "indice_de_charlson",
        "sintoma_9_0", "vasopresores", "estado_mental_alterado", "hipoxemia", 
        "cateter_venoso", "tiempo_ultima"
    ])
    column_mapping = {
        "sintoma_2_0": "sintoma_2.0",
        "sintoma_8_0": "sintoma_8.0",
        "sintoma_9_0": "sintoma_9.0"
    }
    new_patient_data.rename(columns=column_mapping, inplace=True)
    return create_pca_graph(new_patient_data)



if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
 
 
 
 # Escucha en localhost:8050
 # if __name__ == "__main__":
    #app.run_server(debug=True, port=8050) 
