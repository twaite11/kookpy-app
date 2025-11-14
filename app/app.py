import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import kookpy
from datetime import datetime, timedelta
import base64
import numpy as np


# page config setup
st.set_page_config(layout="wide", page_title="Kookpy AI Surf Forecast")

# --- retro theme elements ---
BG_DARK = "#0E1117"
TEXT_LIGHT = "#e0d8ff"        # lavender
GRADIENT_LIGHT = "#B8A2F2"    # Brighter lavender
GRADIENT_DARK = "#8A5AD0"     # Brighter deep purple
BUTTON_BG = "#312A45"         # Dark button background
GRID_LINE_COLOR = "#1f2333"   # greyish dividing lines

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* --- GLOBAL FONT & THEME --- */
    body, .stApp, div, p, span, h1, h2, h3, h4, h5, h6, th, td {{
        font-family: 'VT323', monospace !important;
        color: {TEXT_LIGHT} !important; 
    }}
    
    body, .stApp {{
        background-color: {BG_DARK} !important;
    }}

    /* Titles - Purple Gradient */
    h1, h2, h3 {{
        background: -webkit-linear-gradient(45deg, {GRADIENT_LIGHT}, {GRADIENT_DARK});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }}
    
    /* Ensure markdown text uses the light color */
    .stMarkdown p, .stAlert {{
        color: {TEXT_LIGHT} !important;
    }}
    
    /* STYLING FOR TABS, BUTTONS, INPUTS */
    
    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab"] {{
        background-color: {BUTTON_BG} !important;
        color: {TEXT_LIGHT} !important;
        border: 1px solid {GRADIENT_LIGHT} !important;
        border-radius: 8px !important;
        font-family: 'VT323', monospace !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {GRADIENT_DARK} !important;
        color: {BG_DARK} !important;
        font-weight: 700 !important;
        border-color: {GRADIENT_LIGHT} !important;
    }}

    /* Buttons  */
    [data-testid="stButton"] button, .stDownloadButton button {{
        background-color: {BUTTON_BG} !important;
        color: {TEXT_LIGHT} !important;
        border: 1px solid {GRADIENT_LIGHT} !important;
        font-size: 1.2em !important; 
        padding: 10px 15px !important;
        border-radius: 8px !important;
        transition: all 0.2s;
        font-family: 'VT323', monospace !important; 
    }}
    [data-testid="stButton"] button:hover, .stDownloadButton button:hover {{
        background-color: {GRADIENT_LIGHT} !important;
        color: {BG_DARK} !important;
        border-color: {GRADIENT_DARK} !important;
        box-shadow: 0 0 10px {GRADIENT_LIGHT};
    }}
    [data-testid="stButton"] button p, .stDownloadButton button p {{ 
        font-family: 'VT323', monospace !important;
        color: {TEXT_LIGHT} !important;
    }}
    [data-testid="stButton"] button:hover p, .stDownloadButton button:hover p {{
        color: {BG_DARK} !important;
    }}
    
    /* Text Inputs and Select Boxes */
    .stTextInput > div > div > input, .stSelectbox > div > div, .stSelectbox > div > label {{
        background-color: {BUTTON_BG} !important;
        color: {TEXT_LIGHT} !important;
        border: 1px solid {GRID_LINE_COLOR} !important;
        border-radius: 4px;
        font-family: 'VT323', monospace !important;
        font-size: 1.1em;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# svg icon functions -- youtube video demo's

def create_logo_svg():
    # abstract purple gradient wave-like logo
    return f"""
    <svg width="60" height="60" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:{GRADIENT_LIGHT};stop-opacity:1" />
          <stop offset="100%" style="stop-color:{GRADIENT_DARK};stop-opacity:1" />
        </linearGradient>
      </defs>
      <path d="M10 50 Q30 30 50 50 T90 50" stroke="url(#waveGradient)" stroke-width="10" fill="none" stroke-linecap="round"/>
    </svg>
    """


def create_wave_icon(height_ft):
    # svg for a wave height
    scaled_height = min(1.0, height_ft / 10.0)

    return f"""
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <style>
            @keyframes wave-motion {{
                0% {{ transform: scaleY(1) translateY(0); }}
                50% {{ transform: scaleY(1.2) translateY(-5px); }}
                100% {{ transform: scaleY(1) translateY(0); }}
            }}
            .wave-body {{
                fill: {GRADIENT_LIGHT}; /* Light Lavender */
                animation: wave-motion 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
            }}
        </style>
        <path class="wave-body" d="M0,50 Q25,25 50,50 T100,50" style="transform-origin: 50% 50%; transform: scaleY({0.5 + scaled_height/2});"/>
    </svg>
    """


def create_wind_icon(speed, direction):
    # svg for a wind direction
    animation_duration = max(0.5, 2 - (speed / 30))
    rotation = direction + 180
    return f"""
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <style>
            .wind-arrow {{
                fill: {GRADIENT_DARK}; /* Deep Purple */
                transform-origin: 50% 50%;
                animation: wind-flow 2s infinite linear;
            }}
            @keyframes wind-flow {{
                0% {{ transform: rotate({rotation}deg) scale(1); }}
                50% {{ transform: rotate({rotation}deg) scale(1.1); }}
                100% {{ transform: rotate({rotation}deg) scale(1); }}
            }}
        </style>
        <path class="wind-arrow" d="M50,15 L60,35 L50,30 L40,35 Z" />
        <circle cx="50" cy="50" r="30" fill="none" stroke="{GRADIENT_DARK}" stroke-width="2" />
        <path d="M50,20 L50,30" stroke="{GRADIENT_DARK}" stroke-width="2"/>
        <path d="M50,70 L50,80" stroke="{GRADIENT_DARK}" stroke-width="2"/>
        <path d="M20,50 L30,50" stroke="{GRADIENT_DARK}" stroke-width="2"/>
        <path d="M70,50 L80,50" stroke="{GRADIENT_DARK}" stroke-width="2"/>
    </svg>
    """


def create_viridis_color(normalized_score):
    # generate a hex color
    colors = ['#AA55AA', '#9955CC', '#7755EE', '#5555FF',
              '#3377FF', '#1199FF', '#11BBEE', '#11CCCC']
    index = int(normalized_score * (len(colors) - 1))
    return colors[index]


def create_score_icon(score, max_score=10):
    # svg for a circular score meter with color
    normalized_score = max(0, min(1, score / max_score))
    progress_color = create_viridis_color(normalized_score)
    circumference = 2 * np.pi * 40
    stroke_dashoffset = circumference * (1 - normalized_score)

    return f"""
    <svg width="400" height="400" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="45" stroke="{BUTTON_BG}" stroke-width="10" fill="none"/> 
      <circle cx="50" cy="50" r="45" stroke="{progress_color}" stroke-width="10" fill="none"
              stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_dashoffset}"
              transform="rotate(-90 50 50)"/>
      <text x="50" y="50" font-size="30" fill="{TEXT_LIGHT}" text-anchor="middle" alignment-baseline="middle" style="font-family: 'VT323', monospace;">{score:.1f}</text>
    </svg>
    """


def create_tide_icon():
    # static svg icon for tide data
    return f"""
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M10,50 Q25,40 50,50 T90,50" fill="none" stroke="{GRADIENT_LIGHT}" stroke-width="3" />
        <path d="M50,50 L50,80" stroke="{GRADIENT_LIGHT}" stroke-width="3" />
        <path d="M45,80 L55,80" stroke="{GRADIENT_LIGHT}" stroke-width="3" />
        <path d="M50,50 L40,40 L60,40 L50,50 Z" fill="{GRADIENT_DARK}" />
    </svg>
    """


def image_to_base64(svg_string):
    # converts svg string to a base64-encoded uri
    encoded = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def create_score_legend():
    # generates the ai quality score legend using streamlit components
    st.markdown("### ai wave quality score explained")
    st.markdown("this is a prediction of wave quality on a scale of 1-10. it is a beta feature trained on historical data and is constantly learning.")

    # the color bar gradient copied viridisd
    st.markdown(
        """
        <div style="
            background: linear-gradient(to right, #AA55AA, #9955CC, #7755EE, #5555FF, #3377FF, #1199FF, #11BBEE, #11CCCC);
            height: 20px;
            width: 100%;
            border-radius: 5px;
            margin-right: 15px;
        "></div>
        """,
        unsafe_allow_html=True
    )

    # qualitative scores labels
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"<p style='font-size: 12px; text-align: center; color: #AA55AA; font-family: VT323, monospace;'><b>1</b><br>(bad trip)</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<p style='font-size: 12px; text-align: center; color: #7755EE; font-family: VT323, monospace;'><b>5</b><br>(synth wave)</p>", unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"<p style='font-size: 12px; text-align: center; color: #5555FF; font-family: VT323, monospace;'><b>6</b><br>(rad)</p>", unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"<p style='font-size: 12px; text-align: center; color: #1199FF; font-family: VT323, monospace;'><b>8</b><br>(totally tubular)</p>", unsafe_allow_html=True)
    with col5:
        st.markdown(
            f"<p style='font-size: 12px; text-align: center; color: #11CCCC; font-family: VT323, monospace;'><b>10</b><br>(gnarly)</p>", unsafe_allow_html=True)

# desc bar

def create_description_ui():
    # button and conditional display for the description

    if 'show_description' not in st.session_state:
        st.session_state.show_description = False

    button_label = "click for description?"

    #  toggles state on click
    if st.button(button_label, key='desc_toggle_button'):
        st.session_state.show_description = not st.session_state.show_description

    if st.session_state.show_description:
        # st.markdown
        st.markdown(
            f"""
            <div style="
                border: 2px solid {GRADIENT_LIGHT}; 
                padding: 15px; 
                margin-top: 10px; 
                margin-bottom: 20px;
                background-color: {GRID_LINE_COLOR}; 
                border-radius: 8px;
                font-family: 'VT323', monospace;
                color: {TEXT_LIGHT};
                font-size: 1.1em;
                line-height: 1.2;
                ">
            
            listen up, kook. this isn't your granddad's weather report.
            
            we take real marine data—like swell height, period, wind speed, any effect that big olde' moon has on the ocean...
            and feed it to a (GNN) gnarly neural network trained on decades of historic wave conditions. it's basically a highly-paid psychic dolphin
            
            **how to use this beast:**
            
            1.  **log in:** yeah, we're exclusive. sign up on the main page button below.
            2.  **search:** punch in the name of your favorite california break (or waikiki, we guess) and hit that button.
            3.  **check the score:** the bar chart shows the predicted **ai wave quality score** (1-10) for the next seven days. 
            4.  **if it's 8 or above:** ditch work immediately. don't even finish reading this.
            
            </div>
            """, unsafe_allow_html=True
        )

# CRUD Management Functions
def create_account_management_ui():
    st.subheader("account management")

    # CHANGE PASSWORD FORM
    st.markdown("##### change password")
    with st.container(border=True):
        new_pass1 = st.text_input("new password", type="password", key="new_pass1")
        new_pass2 = st.text_input("confirm new password", type="password", key="new_pass2")

        if st.button("update password", key="update_pass_button"):
            username = st.session_state.username

            # validation check
            if new_pass1 != new_pass2:
                st.error("passwords do not match. try again.")
            elif len(new_pass1) < 6:
                st.error("password must be at least 6 characters long.")
            else:
                if kookpy.user_db.modify_user(username, new_pass1):
                    st.success("password updated successfully! please sign in again.")
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    st.rerun()
                else:
                    st.error("failed to update password.")

    # DELETE ACCOUNT BUTTON
    st.markdown("##### permanently delete account")
    with st.container(border=True):
        st.warning("this action cannot be undone and will delete your user profile.")

        delete_confirm = st.text_input(f"type '{st.session_state.username}' to confirm deletion", key="delete_confirm_input")

        if st.button("delete my account", key="delete_account_button"):
            username = st.session_state.username

            # security validation: user must type their own username
            if delete_confirm == username:
                if kookpy.user_db.delete_user(username):
                    st.error("account successfully deleted. goodbye.")
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    st.rerun()
                else:
                    st.error("deletion failed. contact sysop.")
            else:
                st.error("confirmation text did not match username.")


def login_form():
    # renders the login/signup form i

    # center the login box
    col_l, col_center, col_r = st.columns([1, 4, 1])

    with col_center:
        st.markdown("### access protocol")

        # container for visual grouping
        with st.container(border=True):
            choice = st.radio("mode", ["login", "sign up"], horizontal=True)

            if choice == "login":
                username = st.text_input("username", key="login_user")
                password = st.text_input("password", type="password", key="login_pass")

                if st.button("login", key="login_main_button"):
                    if kookpy.user_db.verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("invalid username or password")

            elif choice == "sign up":
                new_user = st.text_input("new username", key="signup_user")
                new_pass = st.text_input("new password", type="password", key="signup_pass")

                if st.button("sign up", key="signup_main_button"):
                    # basic validation functionality
                    if len(new_user) > 3 and len(new_pass) > 5:
                        if kookpy.user_db.add_user(new_user, new_pass):
                            st.success("account created! please switch to login mode.")
                        else:
                            st.error("username already taken.")
                    else:
                        st.error("username must be > 3 chars, password must be > 5 chars.")


def logout_button():
    # logout button in the top right header column
    if st.button("sign out", key="logout_top_right"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.run_forecast = False # reset app state
        st.rerun()


def manage_account_button():
    if 'show_manage_account' not in st.session_state:
        st.session_state.show_manage_account = False

    if st.button("manage account", key="manage_account_top_right"):
        st.session_state.show_manage_account = not st.session_state.show_manage_account


# main application logic

def main_app():

    col_logo, col_title, col_spacer, col_manage, col_logout = st.columns([1, 5, 2, 2, 2])

    with col_logo:
        st.image(image_to_base64(create_logo_svg()), width=60)

    with col_title:
        st.title(f"kookpy ai surf forecast - logged in as {st.session_state.username}")
        st.markdown("### powered by the open-meteo api and tensorflow")
        st.markdown("---")
        create_description_ui()

    with col_logout:
        # logout button in the top right column
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Vertical spacing
        logout_button()
        manage_account_button()

    st.markdown("---")

    # ACCOUNT MANAGEMENT PANEL
    if st.session_state.show_manage_account:
        create_account_management_ui()
        st.markdown("---")

    # MAIN FORECAST CONTENT
    st.markdown("## data access protocol")

    # user input section
    tabs = st.tabs(["search by name", "select from list"])

    with tabs[0]:
        # unique key assigned to text input
        beach_name_input = st.text_input(
            "enter a beach name:", "laguna beach", key='search_beach_input', help="e.g., laguna beach, huntington beach, waikiki")

        if st.button("get forecast & prediction", type="primary", key='search_button'):
            search_query = st.session_state.search_beach_input # read the current value
            if not search_query:
                st.error("please enter a valid beach name.")
            else:
                st.session_state.run_forecast = True
                st.session_state.beach_name = search_query
                st.rerun() # force immediate update

    with tabs[1]:
        california_beaches = [
            "huntington beach", "malibu", "santa cruz", "la jolla", "trestles",
            "steamer lane", "rincon", "newport beach", "pacifica state beach", "point dume",
            "zuma beach", "el porto", "venice beach", "manhattan beach", "hermosa beach",
            "redondo beach", "torrance beach", "cabrillo beach", "dana point", "san onofre",
            "swami's", "cardiff reef", "ponto beach", "oceanside harbor", "black's beach",
            "del mar", "encinitas", "solana beach", "mission beach", "ocean beach",
            "sunset cliffs", "imperial beach", "fort point", "ocean beach, san francisco",
            "half moon bay", "mavericks", "bolinas", "stinson beach", "montara",
            "cowell's beach", "pleasure point", "capitola", "seabright beach", "manresa state beach",
            "moss landing", "marina state beach", "carmel beach", "asilomar state beach",
            "morro bay", "pismo beach", "avila beach", "cayucos", "cambria",
            "point conception", "jalama beach", "refugio state beach", "el capitan state beach",
            "gaviota state park", "carpinteria", "summerland", "leadbetter beach", "campus point",
            "isla vista", "mondos", "emma wood", "c street, ventura", "silver strand",
            "leo carrillo state park", "el matador state beach", "topanga state beach",
            "surfrider beach", "county line", "zuma", "oxnard shores", "ventura point",
            "rincon point", "pismo state beach", "grover beach", "santa monica state beach",
            "dockweiler beach", "manhattan beach pier", "venice breakwater", "san clemente pier",
            "doheny state beach", "salt creek", "strands beach", "thalia street",
            "brook street", "main beach, laguna", "table rock beach", "aliso beach",
            "laguna niguel", "dana strands", "san clemente", "huntington cliffs",
            "seal beach", "alamitos bay", "belmont shore", "long beach", "point mugu",
            "morro strand state beach", "sunset beach, orange county", "bolsa chica state beach",
            "san elijo state beach"
        ]
        beach_name_select = st.selectbox(
            "select a popular california beach:", california_beaches)
        if st.button("get forecast for selected beach", type="primary"):
            st.session_state.run_forecast = True
            st.session_state.beach_name = beach_name_select
            st.rerun() # force immediate update

    # forecast and prediction display
    if "run_forecast" in st.session_state and st.session_state.run_forecast:
        with st.spinner(f"fetching data and generating prediction for {st.session_state.beach_name}..."):
            # get location coordinates first
            coords = kookpy.geocode_location(st.session_state.beach_name)
            if not coords:
                st.error("could not find coordinates for that location.")
                st.session_state.run_forecast = False
                st.stop()

            # get data using the coordinates
            forecast_df = kookpy.get_surf_forecast_by_name(
                st.session_state.beach_name)

            if forecast_df.empty:
                st.error(
                    "could not find forecast for that location. please try another name or check your internet connection.")
                st.session_state.run_forecast = False
            else:
                try:
                    # ensure the dataframe has the columns needed for prediction
                    required_features = [
                        'swell_wave_height', 'swell_wave_period', 'wind_speed_10m', 'sea_level_height_msl']
                    if not all(feature in forecast_df.columns for feature in required_features):
                        st.error(
                            "forecast data is missing required features for ai prediction.")
                        st.session_state.run_forecast = False
                        st.stop()

                    forecast_df['wave_quality_score'] = forecast_df.apply(
                        lambda row: kookpy.predict_surf_quality(row), axis=1
                    )
                except Exception as e:
                    st.error(
                        f"prediction failed. have you trained your model by running 'model_trainer.py'? error: {e}")
                    st.session_state.run_forecast = False
                    st.stop()

                # get tide data for the next 48 hours to find high/low tides
                tide_data = kookpy.fetch_tide_data(coords['latitude'], coords['longitude'], datetime.now(
                ).date().strftime('%Y-%m-%d'), (datetime.now().date() + timedelta(days=2)).strftime('%Y-%m-%d'))

                forecast_df['swell_wave_height_ft'] = forecast_df['swell_wave_height'] * 3.281

                # generate report data (now meets the report requirement)
                report_df = forecast_df[['time', 'swell_wave_height_ft', 'swell_wave_period', 'wind_speed_10m', 'wave_quality_score']].copy()
                st.session_state.report_df = report_df

                st.success(
                    f"forecast and prediction for {st.session_state.beach_name} ready.")
                st.markdown("---")

                # current conditions summary
                if not forecast_df.empty:
                    st.subheader("current conditions")
                    now_df = forecast_df.iloc[0]

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"**ai quality score**")
                        score_icon_svg = create_score_icon(
                            now_df['wave_quality_score'])
                        st.image(image_to_base64(score_icon_svg), width=200)

                    with col2:
                        st.markdown(f"**current wave height**")
                        st.markdown(
                            f"<p style='font-size: 30px; margin: 0; color: {TEXT_LIGHT}; font-family: VT323, monospace;'>{now_df['swell_wave_height_ft']:.1f} ft</p>", unsafe_allow_html=True)
                        wave_icon_svg = create_wave_icon(
                            now_df['swell_wave_height_ft'])
                        st.image(image_to_base64(wave_icon_svg), width=100)

                    with col3:
                        st.markdown(f"**current wind**")
                        st.markdown(
                            f"<p style='font-size: 30px; margin: 0; color: {TEXT_LIGHT}; font-family: VT323, monospace;'>{now_df['wind_speed_10m']:.1f} km/h</p>", unsafe_allow_html=True)
                        wind_icon_svg = create_wind_icon(
                            now_df['wind_speed_10m'], now_df['wind_direction_10m'])
                        st.image(image_to_base64(wind_icon_svg), width=100)

                    with col4:
                        st.markdown(f"**tide**")
                        if tide_data and 'next_high_tide' in tide_data:
                            st.markdown(
                                f"<p style='font-size: 30px; margin: 0; color: {TEXT_LIGHT}; font-family: VT323, monospace;'>{tide_data['next_high_tide']['height_m'] * 3.281:.1f} ft</p>", unsafe_allow_html=True)
                            tide_data_html = f"<div style='font-size: 14px; color: {TEXT_LIGHT}; font-family: VT323, monospace;'><b>next tides:</b><p style='margin: 0;'>high: {tide_data['next_high_tide']['time']} ({tide_data['next_high_tide']['height_m'] * 3.281:.1f} ft)</p></div>"
                            if 'next_low_tide' in tide_data:
                                tide_data_html += f"<p style='margin: 0;'>low: {tide_data['next_low_tide']['time']} ({tide_data['next_low_tide']['height_m'] * 3.281:.1f} ft)</p>"
                            st.markdown(tide_data_html, unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<p style='font-size: 30px; margin: 0; color: {TEXT_LIGHT};'>n/a</p>", unsafe_allow_html=True)
                            st.write("tide data not available.")
                        tide_icon_svg = create_tide_icon()
                        st.image(image_to_base64(tide_icon_svg), width=100)

                st.markdown("---")
                st.subheader("7-day forecast")

                # report download functionality
                if 'report_df' in st.session_state:
                    st.download_button(
                        label="generate & download 7-day report (csv)",
                        data=st.session_state.report_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"{st.session_state.beach_name.lower().replace(' ', '_')}_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="secondary"
                    )

                # --- ai quality score explanation ---
                create_score_legend()

                # --- visualization ---
                forecast_df_3hr = forecast_df[forecast_df['time'].dt.hour % 3 == 0]

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=(f"swell wave height and predicted quality for {st.session_state.beach_name}", "wind speed forecast", "tide forecast"))

                # --- wave height bar chart ---
                # colorscale from youtube vid
                fig.add_trace(go.Bar(
                    x=forecast_df_3hr['time'],
                    y=forecast_df_3hr['swell_wave_height_ft'],
                    marker_color=forecast_df_3hr['wave_quality_score'],
                    marker_colorscale=[[0, '#AA55AA'], [0.5, '#5555FF'], [1, '#11CCCC']], # Score 1 (Dark Purple) -> Score 10 (Light Cyan)
                    marker_cmin=1,
                    marker_cmax=10,
                    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>wave height: %{y:.2f} ft<br>quality score: %{marker.color:.1f}<extra></extra>",
                    name="wave height",
                    showlegend=False
                ), row=1, col=1)

                # vertical dashed lines for each day and date headers
                dates = pd.to_datetime(forecast_df['time']).dt.date.unique()
                for i, date in enumerate(dates):
                    date_str = date.strftime('%Y-%m-%d')
                    fig.add_vline(x=date_str, line_width=1, line_dash="dash",
                                  line_color=GRADIENT_DARK, opacity=0.5, row=1, col=1)
                    fig.add_vline(x=date_str, line_width=1, line_dash="dash",
                                  line_color=GRADIENT_DARK, opacity=0.5, row=2, col=1)
                    fig.add_vline(x=date_str, line_width=1, line_dash="dash",
                                  line_color=GRADIENT_DARK, opacity=0.5, row=3, col=1)

                    if i < len(dates) - 1:
                        mid_point = date + (dates[i+1] - date) / 2
                        fig.add_annotation(
                            x=mid_point,
                            y=-1.05,
                            text=date.strftime('%b %d'),
                            xref="x",
                            yref="paper",
                            font=dict(color=TEXT_LIGHT, size=16, family="VT323, monospace")
                        )

                # wind speed line chart
                fig.add_trace(go.Scatter(
                    x=forecast_df['time'],
                    y=forecast_df['wind_speed_10m'],
                    mode='lines',
                    name='wind speed (km/h)',
                    line=dict(color=GRADIENT_DARK, dash='dot'),
                    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>wind speed: %{y:.2f} km/h<extra></extra>"
                ), row=2, col=1)

                # tide chart
                fig.add_trace(go.Scatter(
                    x=forecast_df['time'],
                    y=forecast_df['sea_level_height_msl'],
                    mode='lines',
                    name='sea level height',
                    line=dict(color=GRADIENT_LIGHT),
                    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>tide: %{y:.2f} m<extra></extra>"
                ), row=3, col=1)

                high_tides = forecast_df[forecast_df['sea_level_height_msl'] == forecast_df['sea_level_height_msl'].rolling(
                    window=3, center=True).max()].dropna()
                low_tides = forecast_df[forecast_df['sea_level_height_msl'] == forecast_df['sea_level_height_msl'].rolling(
                    window=3, center=True).min()].dropna()

                fig.add_trace(go.Scatter(
                    x=high_tides['time'],
                    y=high_tides['sea_level_height_msl'],
                    mode='markers',
                    name='high tide',
                    marker=dict(symbol='triangle-up', size=10, color=GRADIENT_DARK),
                    hovertemplate="<b>high tide</b><br>date: %{x|%b %d, %I:%M %p}</b><br>height: %{y:.2f} m<extra></extra>"
                ), row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=low_tides['time'],
                    y=low_tides['sea_level_height_msl'],
                    mode='markers',
                    name='low tide',
                    marker=dict(symbol='triangle-down', size=10, color=GRADIENT_LIGHT),
                    hovertemplate="<b>low tide</b><br>date: %{x|%b %d, %I:%M %p}</b><br>height: %{y:.2f} m<extra></extra>"
                ), row=3, col=1)

                fig.update_yaxes(title_text="swell wave height (ft)", row=1, col=1, title_font=dict(family="VT323, monospace"))
                fig.update_yaxes(title_text="wind speed (km/h)", row=2, col=1, title_font=dict(family="VT323, monospace"))
                fig.update_yaxes(title_text="sea level (m)", row=3, col=1, title_font=dict(family="VT323, monospace"))
                fig.update_xaxes(title_text="date and time", row=3, col=1, title_font=dict(family="VT323, monospace"))

                # layout update for the retro theme
                fig.update_layout(hovermode="x unified",
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  font=dict(family="VT323, monospace"),
                                  font_color=TEXT_LIGHT,
                                  xaxis_gridcolor=GRID_LINE_COLOR,
                                  yaxis_gridcolor=GRID_LINE_COLOR,
                                  margin=dict(b=100))
                fig.update_layout(height=1000)

                fig.update_layout(
                    title_text=f"swell wave height and predicted quality for {st.session_state.beach_name}",
                    legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

# --- run application ---

# initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = None

if 'show_manage_account' not in st.session_state:
    st.session_state.show_manage_account = False

# main flow
if st.session_state.logged_in:
    # run the main application content
    main_app()
else:
    # LOGGED OUT VIEW
    col_l, col_center, col_r = st.columns([1, 4, 1])

    with col_center:
        # Header: Logo, Title (Centered)
        col_logo, col_title = st.columns([1, 10])
        with col_logo:
            st.image(image_to_base64(create_logo_svg()), width=60)
        with col_title:
            st.title("welcome to kookpy ai surf forecast")
            st.markdown("### powered by the open-meteo api and tensorflow")

            create_description_ui()
            st.info("please log in or sign up below to view the forecast.")

        login_form()