import streamlit as st
import joblib
import math
from datetime import date, datetime

import folium
import geopandas as gpd
from streamlit_folium import st_folium

import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np

def display_map(seattle_mcpp):
    seattle_geo = folium.GeoJson(
        seattle_mcpp, 
        name="MCPP", 
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        highlight_function=lambda x: {
            'fillOpacity':1
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['neighborho'],
            aliases=['Micro Community: '],
        )
    )

    m = folium.Map(location=(47.62, -122.3), zoom_start=11)
    seattle_geo.add_to(m)

    st_data = st_folium(m, width=625, returned_objects=["last_active_drawing"])

    return st_data

def display_3d_map(seattle_mcpp, seattle_mcpp_complete, year, year_labels, season, season_labels):

    year_selected, month_selected = get_selected_month_year(year, year_labels, season, season_labels)

    time_filtered_complete = seattle_mcpp_complete.loc[
        (seattle_mcpp_complete['Report Year'].isin(year_selected)) & 
        (seattle_mcpp_complete['Report Month'].isin(month_selected))
    ]

    mcpp_count = time_filtered_complete.groupby(by="MCPP")["Count"].sum()
    mcpp_count_df = pd.DataFrame({'MCPP':mcpp_count.index, 'Count':mcpp_count.values})
    complete_df = seattle_mcpp.merge(mcpp_count_df, how="left")

    INITIAL_VIEW_STATE = pdk.ViewState(latitude=47.62, 
                                   longitude=-122.3, 
                                   zoom=10, max_zoom=16, pitch=60, bearing=0, height=800)

    max_cnt = max(complete_df["Count"])
    layers = pdk.Layer(
        "GeoJsonLayer",
        complete_df,
        opacity=1,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        pickable=True,
        get_elevation="Count",
        get_fill_color="[255, 255, (Count / 2000) * 255]"
    )

    st.pydeck_chart(
        pdk.Deck(layers=[layers], 
                initial_view_state=INITIAL_VIEW_STATE, 
                tooltip={"html": "<b>{MCPP}</b><br>Count: {Count}", "style": {"color": "white"}}
        )
    )

def build_filters():
    st.sidebar.title("Filters")

    year_values = range(17)
    year_labels = ["All"] + [*range(2008, 2023, 1)] + ["Future"]
    year = st.sidebar.select_slider(
        "Pick a Year:",
        options=year_values,
        value=0,
        format_func=(lambda x: year_labels[x])
    )

    season_values = range(5)
    season_labels = ["All", "Spring", "Summer", "Fall", "Winter"]

    season = st.sidebar.selectbox(
        "Select a Season:",
        options=season_values,
        index=0,
        format_func=(lambda x: season_labels[x])
    )

    is_pred = year_labels[year] == "Future"
    is_3d_map = st.sidebar.checkbox("Enable 3D Map", disabled=is_pred)

    return year, year_labels, season, season_labels, is_3d_map

def get_selected_month_year(year, year_labels, season, season_labels):
    year_selected = []
    if year_labels[year] == "All":
        year_selected = [*range(2008, 2023, 1)]
    elif year_labels[year] == "Future":
        year_selected = []
    else:
        year_selected = [year_labels[year]]

    month_selected = []
    if season_labels[season] == "All":
        month_selected = [*range(1, 12)]
    elif season_labels[season] == "Spring":
        month_selected = [3, 4, 5]
    elif season_labels[season] == "Summer":
        month_selected = [6, 7, 8]
    elif season_labels[season] == "Fall":
        month_selected = [9, 10, 11]
    elif season_labels[season] == "Winter":
        month_selected = [12, 1, 2]

    return year_selected, month_selected

def show_metric(crime_df, year, year_labels, season, season_labels, st_data):
    year_selected, month_selected = get_selected_month_year(year, year_labels, season, season_labels)

    mcpp_selected = crime_df["MCPP"].unique()
    if st_data["last_active_drawing"]:
        mcpp_selected = [st_data["last_active_drawing"]["properties"]["neighborho"]]

    time_filtered_offense = crime_df.loc[
        (crime_df['Report Year'].isin(year_selected)) & 
        (crime_df['Report Month'].isin(month_selected)) &
        (crime_df['MCPP'].isin(mcpp_selected))
    ]

    offense_cnt = time_filtered_offense.shape[0]
    st.metric("Total offense report count", offense_cnt)

    return time_filtered_offense

def draw_donut_chart(time_filtered_offense):
    unique_offense_cnts = time_filtered_offense["Offense"].value_counts()
    offense_cnts_df = pd.DataFrame({'Offense':unique_offense_cnts.index, 'Count':unique_offense_cnts.values})
    # draw donut chart
    chart = alt.Chart(
            offense_cnts_df
        ).mark_arc(innerRadius=50, stroke="#fff").encode(
            alt.Theta("Count:Q").stack(True),
            alt.Radius("Count").scale(type="sqrt", zero=True, rangeMin=20),
            alt.Color("Offense:N", sort=['Count']),
            order='Count:Q'
        )
    # draw overlay text
    text = chart.transform_joinaggregate(
            TotalCnt="sum(Count)"
        ).transform_calculate(
            PercentOfTotal="datum.Count / datum.TotalCnt"
        ).mark_text(
            radiusOffset=15
        ).encode(
            # alt.Text(field="PercentOfTotal", type="quantitative", format=".1%")
            text=alt.condition(
                alt.datum.PercentOfTotal > 0.005, 
                alt.Text(field="PercentOfTotal", type="quantitative", format=".1%"), 
                alt.value('')
            )
        )

    st.altair_chart((chart + text), theme="streamlit", use_container_width=True)

def crime_rate_prediction(mcpp_census_df, selected_mcpp_index, selected_season, crime_rate_model):
    userInput = [selected_mcpp_index, selected_season]

    # gotta drop MCPP column
    mcpp_census_df = mcpp_census_df.drop(columns=['MCPP'])
    mcpp_data = mcpp_census_df.loc[userInput[0]].values.tolist()
    X_test_modelInput = np.append(userInput, mcpp_data)
    pred_output = crime_rate_model.predict(X_test_modelInput.reshape(1, -1))

    if pred_output < 0:
        return 0
    else:
        return pred_output

def likelihood_prediction(selected_season, selected_mcpp_index, likelihood_model):
    model_input = [[selected_season, selected_mcpp_index]]
    predicted_top_offense = likelihood_model.predict(model_input)[0]
    predicted_offense_prob = likelihood_model.predict_proba(model_input)[0]

    return predicted_top_offense, max(predicted_offense_prob) * 100

# cache data if possible
@st.cache_data
def load_data():
    crime_df = pd.read_csv("./data/SPD_Crime_Data__2008-Present_20231028.csv")
    crime_df["Report Year"] = pd.to_datetime(crime_df["Report DateTime"]).dt.year
    crime_df["Report Month"] = pd.to_datetime(crime_df["Report DateTime"]).dt.month

    seattle_mcpp = gpd.read_file('./data/Seattle_MCPP/MCPP_WM.shp').to_crs("EPSG:4326")
    seattle_mcpp["MCPP"] = seattle_mcpp["neighborho"]

    seattle_mcpp_complete = gpd.read_file("./data/seattle_crime_complete_mcp.json")

    # Loading model 
    crime_rate_model = joblib.load('./model/reg_gb_crimeRate_gridCV_SEASON.joblib')
    crime_rate_range_df = pd.DataFrame({
        "Range": ["Low", "Medium", "Medium High", "High"],
        "Interval": ["14.5", "22.2", "32.2", "59.3"]
    })

    mcpp_census_df = pd.read_csv("./data/mcpp_2010_PopulationDemographic_estimate_df.csv")
    cleaned_mcpp_census_df = mcpp_census_df.drop(columns=['Unnamed: 0'])

    likelihood_model = joblib.load('./model/CLF_grid_search_RF_SEASON.joblib')

    # Crime forecast for 2024
    future_crime_2024 = pd.read_csv('./data/futurecrime.csv')

    return crime_df, seattle_mcpp, seattle_mcpp_complete, crime_rate_model, crime_rate_range_df, cleaned_mcpp_census_df, likelihood_model, future_crime_2024

def main():

    st.set_page_config(
        page_title="CS6650 final project - Seattle Crime",
        page_icon=":world_map:️",
        layout="wide",
    )
    st.title("Seattle Crime Dataset Demo")

    # load data
    crime_df, seattle_mcpp, seattle_mcpp_complete, crime_rate_model, crime_rate_range_df, mcpp_census_df, likelihood_model, future_crime_2024 = load_data()
    # build filters on sidebar
    year, year_labels, season, season_labels, is_3d_map = build_filters()

    is_prediction = year_labels[year] == "Future"

    if is_3d_map and not is_prediction:
        display_3d_map(seattle_mcpp, seattle_mcpp_complete, year, year_labels, season, season_labels)

    else:
        left, right = st.columns(2)

        with left:

            st_data = display_map(seattle_mcpp)

            if st.button("reset community selection"):
                st_data["last_active_drawing"] = None

        with right:
            # st.write("Year:", year_labels[year])
            # st.write("Season:", season_labels[season])

            comm_selected = "All"
            if st_data["last_active_drawing"]:
                comm_selected = st_data["last_active_drawing"]["properties"]["neighborho"]
            st.subheader(f'{season_labels[season]} season, {year_labels[year]} year in {comm_selected} community: ')

            if is_prediction and comm_selected != "All" and season_labels[season] != "All":
                # prediction
                season_encode = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
                offense_encode = ['Assault and Threatening Behavior', 
                                  'Burglary and Trespassing', 
                                  'Drug-Related Offenses',
                                  'Fraud and Financial Crimes',
                                  'Miscellaneous and Less Frequent Crimes',
                                  'Other Thefts',
                                  'Robbery and Serious Theft-Related Crimes', 
                                  'Theft from Vehicles and Related Crimes', 
                                  'Vandalism and Property Damage', 
                                  'Weapon, Public Order, and Other Violations']

                selected_mcpp_index = mcpp_census_df.index[mcpp_census_df['MCPP'] == comm_selected].values[0]
                selected_season = season_encode[season_labels[season]]

                # crime rate prediction
                pred_output = crime_rate_prediction(mcpp_census_df, selected_mcpp_index, selected_season, crime_rate_model)
                st.metric("Crime Rate Prediction(per 1000 people):", "{:.2f}".format(pred_output.flatten()[0]))

                st.dataframe(crime_rate_range_df)

                st.divider()

                # likelihood prediction
                predicted_top_offense, predicted_top_offense_prob = likelihood_prediction(selected_season, selected_mcpp_index, likelihood_model)
                st.metric("Top Offense Prediction:", offense_encode[predicted_top_offense], "With {:.2f}% chance".format(predicted_top_offense_prob))

                st.divider()

                # 2024 forecast
                if st_data["last_active_drawing"]:
                    mcpp_selected = [st_data["last_active_drawing"]["properties"]["neighborho"]]

                time_filtered_forecast = future_crime_2024.loc[
                    (future_crime_2024['season'].isin([season_labels[season]])) &
                    (future_crime_2024['MCPP'].isin(mcpp_selected))
                ]

                forecast_cnt = time_filtered_forecast['PredictedCrimeCount']
                st.metric("Total crime count forecast for 2024: ", forecast_cnt)

            elif is_prediction:
                st.text('Please select a Community or Season!');
            else:
                # show data
                time_filtered_offense = show_metric(crime_df, year, year_labels, season, season_labels, st_data)

                # draw the donut chart
                draw_donut_chart(time_filtered_offense)


if __name__ == "__main__":
    main()
