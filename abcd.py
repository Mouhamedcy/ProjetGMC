import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from io import StringIO
import base64

# Configuration de la page
st.set_page_config(
    page_title="Explorateur de Données",
    page_icon="📊",
    layout="wide"
)


# Fonction pour générer des données d'exemple
def get_sample_data(sample_name):
    if sample_name == "population":
        data = """Pays,Population,Continent,Croissance,Espérance de vie
France,67000000,Europe,0.3,82.5
Japon,126000000,Asie,-0.2,84.2
États-Unis,329000000,Amérique du Nord,0.6,78.9
Brésil,211000000,Amérique du Sud,0.8,75.7
Nigéria,200000000,Afrique,2.6,54.3
Australie,25000000,Océanie,1.2,83.4
Inde,1380000000,Asie,1.0,69.4
Chine,1400000000,Asie,0.3,76.1
Allemagne,83000000,Europe,0.2,81.2
Égypte,100000000,Afrique,1.9,71.8"""
        return pd.read_csv(StringIO(data))

    elif sample_name == "cars":
        data = """Modèle,Marque,Année,Prix,Carburant,Puissance,Consommation
Clio,Renault,2020,15000,Essence,90,5.2
Golf,Volkswagen,2019,22000,Diesel,115,4.8
Model 3,Tesla,2021,43000,Électrique,350,0
A3,Audi,2018,28000,Essence,150,6.1
308,Peugeot,2020,19500,Diesel,130,4.5
Civic,Honda,2019,23000,Essence,125,5.8
Corolla,Toyota,2021,24500,Hybride,122,3.9
Zoe,Renault,2020,32000,Électrique,135,0
Captur,Renault,2021,18900,Diesel,115,4.3
Panda,Fiat,2018,11000,Essence,85,5.5"""
        return pd.read_csv(StringIO(data))

    elif sample_name == "sales":
        data = """Trimestre,Année,Produit,Région,Ventes,Coût,Profit
T1,2020,Ordinateurs,Europe,1250,875000,375000
T1,2020,Téléphones,Europe,2800,1400000,840000
T1,2020,Accessoires,Europe,4200,630000,252000
T1,2020,Ordinateurs,Amérique,950,665000,285000
T1,2020,Téléphones,Amérique,1800,900000,360000
T1,2020,Accessoires,Amérique,2500,375000,150000
T2,2020,Ordinateurs,Europe,1400,980000,420000
T2,2020,Téléphones,Europe,3100,1550000,930000
T2,2020,Accessoires,Europe,3800,570000,228000
T2,2020,Ordinateurs,Amérique,1050,735000,315000
T2,2020,Téléphones,Amérique,2100,1050000,420000
T2,2020,Accessoires,Amérique,2900,435000,174000
T3,2020,Ordinateurs,Europe,1100,770000,330000
T3,2020,Téléphones,Europe,2500,1250000,750000
T3,2020,Accessoires,Europe,3200,480000,192000
T3,2020,Ordinateurs,Amérique,1200,840000,360000
T3,2020,Téléphones,Amérique,2300,1150000,460000
T3,2020,Accessoires,Amérique,3100,465000,186000
T4,2020,Ordinateurs,Europe,1800,1260000,540000
T4,2020,Téléphones,Europe,3800,1900000,1140000
T4,2020,Accessoires,Europe,5000,750000,300000
T4,2020,Ordinateurs,Amérique,1650,1155000,495000
T4,2020,Téléphones,Amérique,2900,1450000,580000
T4,2020,Accessoires,Amérique,4200,630000,252000"""
        return pd.read_csv(StringIO(data))

    return None


# Fonction pour télécharger les données en CSV
def get_download_link(df, filename="donnees.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger les données (CSV)</a>'
    return href


# Fonction pour créer les visualisations
def create_visualization(df, chart_type, x_column, y_column, group_by=None):
    if chart_type == "bar":
        if group_by and group_by != "Aucun":
            fig = px.bar(df, x=x_column, y=y_column, color=group_by, barmode="group",
                         title=f"{y_column} par {x_column}")
        else:
            fig = px.bar(df, x=x_column, y=y_column, title=f"{y_column} par {x_column}")

    elif chart_type == "line":
        if group_by and group_by != "Aucun":
            fig = px.line(df, x=x_column, y=y_column, color=group_by, markers=True,
                          title=f"Évolution de {y_column} par {x_column}")
        else:
            fig = px.line(df, x=x_column, y=y_column, markers=True,
                          title=f"Évolution de {y_column} par {x_column}")

    elif chart_type == "scatter":
        if group_by and group_by != "Aucun":
            fig = px.scatter(df, x=x_column, y=y_column, color=group_by,
                             title=f"Relation entre {x_column} et {y_column}")
        else:
            fig = px.scatter(df, x=x_column, y=y_column,
                             title=f"Relation entre {x_column} et {y_column}")

    elif chart_type == "pie":
        if pd.api.types.is_numeric_dtype(df[y_column]):
            # Aggréger les données pour le camembert
            pie_data = df.groupby(x_column)[y_column].sum().reset_index()
            fig = px.pie(pie_data, values=y_column, names=x_column,
                         title=f"Répartition de {y_column} par {x_column}")
        else:
            # Compter les occurrences si la colonne Y n'est pas numérique
            counts = df[x_column].value_counts().reset_index()
            counts.columns = [x_column, 'Count']
            fig = px.pie(counts, values='Count', names=x_column,
                         title=f"Répartition de {x_column}")

    elif chart_type == "box":
        if group_by and group_by != "Aucun":
            fig = px.box(df, x=x_column, y=y_column, color=group_by,
                         title=f"Distribution de {y_column} par {x_column}")
        else:
            fig = px.box(df, x=x_column, y=y_column,
                         title=f"Distribution de {y_column} par {x_column}")

    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_column, title=f"Distribution de {x_column}")

    else:
        st.error("Type de graphique non supporté")
        return None

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column if chart_type != "histogram" and chart_type != "pie" else "",
        height=600,
    )

    return fig


# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #5D5CDE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stAlert > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stFileUploadDropzone"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown('<h1 class="main-header">📊 Explorateur de Données</h1>', unsafe_allow_html=True)

# Initialisation de la session state si nécessaire
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'file_format' not in st.session_state:
    st.session_state.file_format = None

# Sidebar pour le chargement des données
with st.sidebar:
    st.markdown('<h2 class="section-header">Importer des données</h2>', unsafe_allow_html=True)

    # Option 1: Télécharger un fichier
    uploaded_file = st.file_uploader("Télécharger un fichier CSV ou JSON", type=["csv", "json"])

    # Option 2: Utiliser un exemple
    st.markdown("### Ou utiliser un exemple")
    sample_option = st.selectbox(
        "Sélectionner un exemple",
        ["", "population", "cars", "sales"],
        format_func=lambda x: {
            "": "Choisir un exemple",
            "population": "Démographie par pays",
            "cars": "Voitures (caractéristiques)",
            "sales": "Ventes trimestrielles"
        }.get(x, x)
    )

    # Réinitialiser les données
    if st.button("Réinitialiser les données"):
        st.session_state.data = None
        st.session_state.file_name = None
        st.session_state.file_format = None
        st.experimental_rerun()

# Chargement des données
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            file_format = "CSV"
        elif file_name.endswith('.json'):
            df = pd.read_json(uploaded_file)
            file_format = "JSON"
        else:
            st.error("Format de fichier non supporté. Veuillez utiliser CSV ou JSON.")
            df = None
            file_format = None

        if df is not None:
            st.session_state.data = df
            st.session_state.file_name = file_name
            st.session_state.file_format = file_format
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")

# Chargement des exemples
elif sample_option:
    try:
        df = get_sample_data(sample_option)
        if df is not None:
            st.session_state.data = df
            st.session_state.file_name = f"exemple_{sample_option}.csv"
            st.session_state.file_format = "CSV (exemple)"
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'exemple: {str(e)}")

# Affichage des données si disponibles
if st.session_state.data is not None:
    df = st.session_state.data

    # Section d'information
    st.markdown('<h2 class="section-header">Aperçu des données</h2>', unsafe_allow_html=True)

    # Informations sur le jeu de données
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Format", st.session_state.file_format)

    # Lien de téléchargement
    st.markdown(get_download_link(df), unsafe_allow_html=True)

    # Aperçu des données
    with st.expander("Aperçu des données", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Statistiques descriptives
    with st.expander("Statistiques descriptives", expanded=False):
        # Filtrer les colonnes numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("Aucune colonne numérique trouvée pour les statistiques.")

    # Section de visualisation
    st.markdown('<h2 class="section-header">Créer une visualisation</h2>', unsafe_allow_html=True)

    # Contrôles pour la visualisation
    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.selectbox(
            "Type de graphique",
            ["bar", "line", "scatter", "pie", "box", "histogram"],
            format_func=lambda x: {
                "bar": "Histogramme",
                "line": "Ligne",
                "scatter": "Nuage de points",
                "pie": "Camembert",
                "box": "Boîte à moustaches",
                "histogram": "Distribution"
            }.get(x, x)
        )

    with col2:
        # Liste toutes les colonnes comme options
        columns = df.columns.tolist()

        if chart_type == "histogram":
            # Pour l'histogramme, on n'a besoin que de l'axe X
            x_column = st.selectbox("Axe X (variable à analyser)", columns)
            y_column = None
            group_by = None
        elif chart_type == "pie":
            # Pour le camembert, on a besoin de la catégorie et de la valeur
            x_column = st.selectbox("Catégories", columns)
            y_column = st.selectbox("Valeurs", [col for col in columns if col != x_column])
            group_by = None
        else:
            # Pour les autres graphiques
            x_column = st.selectbox("Axe X", columns)
            y_column_options = [col for col in columns if col != x_column]
            if y_column_options:
                y_column = st.selectbox("Axe Y", y_column_options)
            else:
                y_column = None

    # Option de groupement pour les graphiques pertinents
    if chart_type in ["bar", "line", "scatter", "box"] and len(columns) > 2:
        group_by_options = ["Aucun"] + [col for col in columns if col != x_column and col != y_column]
        group_by = st.selectbox("Grouper par", group_by_options)
        if group_by == "Aucun":
            group_by = None
    else:
        group_by = None

    # Créer et afficher la visualisation
    if st.button("Générer la visualisation"):
        if chart_type == "histogram":
            fig = create_visualization(df, chart_type, x_column, None)
        elif y_column:
            fig = create_visualization(df, chart_type, x_column, y_column, group_by)
        else:
            st.error("Veuillez sélectionner des colonnes valides pour la visualisation.")
            fig = None

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Exploration interactive
    st.markdown('<h2 class="section-header">Exploration interactive</h2>', unsafe_allow_html=True)

    # Filtrage des données
    with st.expander("Filtrer les données", expanded=False):
        st.write("Sélectionnez des filtres pour les données:")

        # Créer jusqu'à 3 filtres
        for i in range(3):
            cols = st.columns([3, 2, 3])
            with cols[0]:
                filter_col = st.selectbox(
                    f"Colonne {i + 1}",
                    [""] + columns,
                    key=f"filter_col_{i}"
                )

            if filter_col and filter_col in df.columns:
                col_data = df[filter_col]
                with cols[1]:
                    if pd.api.types.is_numeric_dtype(col_data):
                        filter_type = st.selectbox(
                            "Type de filtre",
                            ["range", "==", ">", "<", ">=", "<="],
                            key=f"filter_type_{i}"
                        )
                    else:
                        filter_type = st.selectbox(
                            "Type de filtre",
                            ["valeurs", "contient"],
                            key=f"filter_type_{i}"
                        )

                with cols[2]:
                    if filter_type == "range" and pd.api.types.is_numeric_dtype(col_data):
                        min_val, max_val = float(col_data.min()), float(col_data.max())
                        filter_value = st.slider(
                            "Plage",
                            min_val, max_val,
                            (min_val, max_val),
                            key=f"filter_value_{i}"
                        )
                    elif filter_type == "valeurs" and not pd.api.types.is_numeric_dtype(col_data):
                        unique_values = col_data.unique().tolist()
                        filter_value = st.multiselect(
                            "Valeurs",
                            unique_values,
                            key=f"filter_value_{i}"
                        )
                    elif filter_type == "contient" and not pd.api.types.is_numeric_dtype(col_data):
                        filter_value = st.text_input(
                            "Contient",
                            key=f"filter_value_{i}"
                        )
                    else:
                        if pd.api.types.is_numeric_dtype(col_data):
                            filter_value = st.number_input(
                                "Valeur",
                                value=float(col_data.mean()),
                                key=f"filter_value_{i}"
                            )
                        else:
                            filter_value = st.text_input(
                                "Valeur",
                                key=f"filter_value_{i}"
                            )

        # Appliquer les filtres
        if st.button("Appliquer les filtres"):
            filtered_df = df.copy()

            for i in range(3):
                filter_col = st.session_state.get(f"filter_col_{i}")
                filter_type = st.session_state.get(f"filter_type_{i}")
                filter_value = st.session_state.get(f"filter_value_{i}")

                if filter_col and filter_col in df.columns and filter_type and filter_value is not None:
                    if filter_type == "range":
                        filtered_df = filtered_df[(filtered_df[filter_col] >= filter_value[0]) &
                                                  (filtered_df[filter_col] <= filter_value[1])]
                    elif filter_type == "==":
                        filtered_df = filtered_df[filtered_df[filter_col] == filter_value]
                    elif filter_type == ">":
                        filtered_df = filtered_df[filtered_df[filter_col] > filter_value]
                    elif filter_type == "<":
                        filtered_df = filtered_df[filtered_df[filter_col] < filter_value]
                    elif filter_type == ">=":
                        filtered_df = filtered_df[filtered_df[filter_col] >= filter_value]
                    elif filter_type == "<=":
                        filtered_df = filtered_df[filtered_df[filter_col] <= filter_value]
                    elif filter_type == "valeurs" and isinstance(filter_value, list):
                        filtered_df = filtered_df[filtered_df[filter_col].isin(filter_value)]
                    elif filter_type == "contient" and isinstance(filter_value, str):
                        filtered_df = filtered_df[
                            filtered_df[filter_col].astype(str).str.contains(filter_value, case=False)]

            st.write(f"Résultat: {len(filtered_df)} lignes sur {len(df)} correspondent aux filtres")
            st.dataframe(filtered_df, use_container_width=True)

            # Option pour utiliser le dataframe filtré
            if st.button("Utiliser ce jeu de données filtré"):
                st.session_state.data = filtered_df
                st.experimental_rerun()
else:
    # Message de démarrage
    st.info("👈 Commencez par importer vos données depuis le menu latéral.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Instructions
        1. Importez des données CSV ou JSON depuis le menu latéral
        2. Explorez les données dans le tableau et les statistiques
        3. Créez différentes visualisations
        4. Filtrez les données selon vos besoins
        """)

    with col2:
        st.markdown("""
        ### Fonctionnalités disponibles
        - Plusieurs types de graphiques
        - Filtrage interactif des données
        - Groupement par catégories
        - Statistiques descriptives
        - Téléchargement des données
        """)