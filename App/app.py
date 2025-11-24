import streamlit as st
import joblib
from Logisticregression import classifyData
# Charger le modèle sauvegardé
model = joblib.load('logistic_regression_model.pkl')

# Définir les couleurs pour les résultats
colors = {
    "non_haineux": "#4CAF50",  # Vert
    "haineux": "#F44336"       # Rouge
}

# Ajouter du CSS pour améliorer l'interface
st.markdown(
    """
    <style>
    .ellipse-container {
        background-color: #FFF3E0;
        border-radius: 50%;
        padding: 50px;
        text-align: center;
        margin: auto;
        width: 80%;
    }
    .header {
        background-color: orange;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .user-info, .message-analysis, .rating-section {
        background-color: #FAFAFA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        margin: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45A049;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #F8F9FA;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        width: 100%;
    }
    .rating-section {
        background-color: yellow;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Conteneur principal avec ellipse
st.markdown("<div class='ellipse-container'>", unsafe_allow_html=True)

# Titre et message de bienvenue
st.markdown(
    """
    <div class='header'>
        <h1>Bienvenue sur l'application de détection de messages haineux</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

# Disposition en colonnes
col1, col2 = st.columns(2)

# Colonne de gauche : Informations utilisateur et options
with col1:
    st.markdown("<div class='user-info'>", unsafe_allow_html=True)
    st.header("Informations Utilisateur")
    full_name = st.text_input("Nom Complet :")
    address = st.text_input("Adresse :")
    
    if st.button('Analyser un message', key='analyser'):
        st.session_state.analyse = True
    if st.button('Quitter', key='quitter'):
        st.session_state.analyse = False
    st.markdown("</div>", unsafe_allow_html=True)

# Colonne de droite : Analyse de message
with col2:
    st.markdown("<div class='message-analysis'>", unsafe_allow_html=True)
    st.header("Analyse de Message")
    if 'analyse' in st.session_state and st.session_state.analyse:
        message = st.text_area("Entrez un message :")
        if st.button('Analyser', key='analyze_message'):
            if message:
                prediction = classifyData(message)
                if prediction == 1:
                    st.markdown(f"<h2 style='color: {colors['haineux']}'>Message haineux</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='color: {colors['non_haineux']}'>Message non-haineux</h2>", unsafe_allow_html=True)
            else:
                st.warning("Veuillez entrer un message à analyser.")
    elif 'analyse' in st.session_state and not st.session_state.analyse:
        st.write(f"Merci d'avoir utilisé notre application, {full_name} !")
        st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

# Zone de notation au fond de l'interface
st.markdown(
    """
    <div class='rating-section'>
        <h2>Notez notre application sur 5 :</h2>
    </div>
    """, 
    unsafe_allow_html=True
)

rating = st.slider("Votre note :", 0, 5, key="rating")

# Fermer le conteneur principal
st.markdown("</div>", unsafe_allow_html=True)