import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd

st.set_page_config(page_title="Find your ideal neighborhood - Paris", layout="wide")

@st.cache_data
def load_data():
    data_iris = pd.read_csv("data_agg_iris.csv")
    data_iris["code_iris"] = data_iris["code_iris"].astype(str).str.replace(".0", "", regex=False)
    try:
        iris_geo = gpd.read_file("iris.shp", encoding="utf-8")
    except:
        iris_geo = gpd.read_file("iris.shp", encoding="latin-1")
    iris_geo["code_iris"] = iris_geo["code_iris"].astype(str).str.replace(".0", "", regex=False)
    iris_geo = iris_geo[iris_geo["dep"] == "75"]
    iris_geo["nom_iris"] = iris_geo["nom_iris"].str.encode("latin-1").str.decode("utf-8", errors="replace")
    iris_merged = iris_geo.merge(
        data_iris[["code_iris", "prix_m2_median", "arrondissement"]],
        on="code_iris", how="left"
    )
    return data_iris, iris_merged

data_iris, iris_merged = load_data()

if "recherche_faite" not in st.session_state:
    st.session_state.recherche_faite = False
if "iris_selectionnes" not in st.session_state:
    st.session_state.iris_selectionnes = []
if "resultats" not in st.session_state:
    st.session_state.resultats = []
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

def reset_all():
    st.session_state.recherche_faite = False
    st.session_state.iris_selectionnes = []
    st.session_state.resultats = []
    st.session_state.reset_counter += 1

cols_model = [
    "Animation", "Commerces et services de proximité", "Culture et loisirs",
    "Enseignement", "Espaces verts", "Santé", "Sport", "Tourisme", "Transports"
]

st.title("Find your ideal neighborhood in Paris")
st.caption("Select your preferences and discover the 5 neighborhoods that best match your profile.")

col1, col2 = st.columns([1, 2.5])

with col1:
    with st.form(key=f"form_{st.session_state.reset_counter}"):
        st.subheader("Your criteria")

        prix_max = st.slider(
            "Maximum budget (€/m²)",
            min_value=int(data_iris["prix_m2_median"].min()),
            max_value=int(data_iris["prix_m2_median"].max()),
            value=int(data_iris["prix_m2_median"].max()),
            step=100,
            format="%d €"
        )

        st.markdown("**Importance of each category** *(0 = not important, 1 = very important)*")

        animation = st.slider("Nightlife & restaurants", 0.0, 1.0, 0.5, step=0.1)
        commerce = st.slider("Shops & proximity services", 0.0, 1.0, 0.5, step=0.1)
        culture = st.slider("Culture & leisure", 0.0, 1.0, 0.5, step=0.1)
        enseignement = st.slider("Education", 0.0, 1.0, 0.5, step=0.1)
        espaces_verts = st.slider("Green spaces", 0.0, 1.0, 0.5, step=0.1)
        sante = st.slider("Health services", 0.0, 1.0, 0.5, step=0.1)
        sport = st.slider("Sport facilities", 0.0, 1.0, 0.5, step=0.1)
        tourisme = st.slider("Tourism", 0.0, 1.0, 0.5, step=0.1)
        transport = st.slider("Public transport", 0.0, 1.0, 0.5, step=0.1)

        bouton = st.form_submit_button("Search", use_container_width=True)

    st.button("Reset", on_click=reset_all, use_container_width=True)

    if bouton:
        data_filtre = data_iris[data_iris["prix_m2_median"] <= prix_max].copy()

        if data_filtre.empty:
            st.warning("No neighborhood matches this budget. Please try a higher budget.")
        else:
            X = data_filtre[cols_model].values
            knn = NearestNeighbors(n_neighbors=min(5, len(data_filtre)), metric="euclidean")
            knn.fit(X)

            profil = [[animation, commerce, culture, enseignement, espaces_verts, sante, sport, tourisme, transport]]
            distances, indices = knn.kneighbors(profil)

            resultats = []
            iris_selectionnes = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                code = data_filtre["code_iris"].iloc[idx]
                ardt = data_filtre["arrondissement"].iloc[idx]
                prix = data_filtre["prix_m2_median"].iloc[idx]
                nom_row = iris_merged[iris_merged["code_iris"] == code]
                nom = nom_row["nom_iris"].values[0] if not nom_row.empty else code
                iris_selectionnes.append(code)
                resultats.append({
                    "Rank": i + 1,
                    "Neighborhood": nom,
                    "Arrondissement": int(ardt),
                    "Median price (€/m²)": f"{int(prix):,} €".replace(",", " ")
                })

            st.session_state.iris_selectionnes = iris_selectionnes
            st.session_state.resultats = resultats
            st.session_state.recherche_faite = True

    if st.session_state.recherche_faite and st.session_state.resultats:
        st.subheader("Results")
        df_resultats = pd.DataFrame(st.session_state.resultats)
        st.dataframe(df_resultats, use_container_width=True, hide_index=True)

with col2:
    carte = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="cartodbpositron")

    if st.session_state.recherche_faite and st.session_state.iris_selectionnes:
        iris_recommandes = iris_merged[iris_merged["code_iris"].isin(st.session_state.iris_selectionnes)]
        iris_autres = iris_merged[~iris_merged["code_iris"].isin(st.session_state.iris_selectionnes)]

        folium.GeoJson(
            iris_autres.to_json(),
            style_function=lambda x: {
                "fillColor": "#cccccc",
                "color": "#999999",
                "weight": 0.5,
                "fillOpacity": 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nom_iris", "prix_m2_median"],
                aliases=["Neighborhood", "Median price (€/m²)"],
                localize=True
            )
        ).add_to(carte)

        folium.GeoJson(
            iris_recommandes.to_json(),
            style_function=lambda x: {
                "fillColor": "#e63946",
                "color": "#c1121f",
                "weight": 2,
                "fillOpacity": 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nom_iris", "prix_m2_median"],
                aliases=["Neighborhood", "Median price (€/m²)"],
                localize=True
            )
        ).add_to(carte)

    else:
        folium.Choropleth(
            geo_data=iris_merged.to_json(),
            data=iris_merged,
            columns=["code_iris", "prix_m2_median"],
            key_on="feature.properties.code_iris",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name="Median price per m² by IRIS"
        ).add_to(carte)

        folium.GeoJson(
            iris_merged.to_json(),
            style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
            tooltip=folium.GeoJsonTooltip(
                fields=["nom_iris", "prix_m2_median"],
                aliases=["Neighborhood", "Median price (€/m²)"],
                localize=True
            )
        ).add_to(carte)

    st_folium(carte, width=None, use_container_width=True, height=600)
