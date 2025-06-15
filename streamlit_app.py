# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 2025 16:45:56

@author: Lakshmi Muthukumar

"""
# Streamlit library
import streamlit as st
import base64
from streamlit import components 
from streamlit_ketcher import st_ketcher
import time

# Importing generic libraries
import csv
import os
import numpy as np
import pandas as pd
import pickle
import IPython
import warnings

# Importing rdkit Libraries
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, PandasTools, MACCSkeys, AtomPairs, rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")

# Pre-processing Libraries
#from imblearn.pipeline import make_pipeline
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
#from collections import Counter
#from sklearn.svm import SVC
#from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,matthews_corrcoef, confusion_matrix ,make_scorer
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from itertools import product
#from collections import Counter
#from imblearn.under_sampling import RandomUnderSampler
#from sklearn.ensemble import VotingClassifier

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background2.jpg') 


st.write("""
# SMILES Endocrine Disruptors App
This app predicts the label of Endocrine Disruptors from input **SMILES** in its notation.
Data used for building the model here, is obtained from the [**paper**].
""")

st.write("""
#### \t Upload smiles notation of the molecule here...
""")

def from_smiles_to_csv(molecule):
    #print(molecule)
    #df = pd.DataFrame(molecule, columns=['smiles'])
    df = pd.DataFrame(molecule, columns = ['smiles'])
    PandasTools.AddMoleculeColumnToFrame(df,'smiles','mol')
    df_maccs = []
    for mol in df['mol']:
        # generate bitvector object
        maccs_bitvector = MACCSkeys.GenMACCSKeys(mol)
        # create an empty array
        arr = np.zeros((0,), dtype=np.int8)
        # convert the RDKit explicit vectors into numpy arrays
        DataStructs.ConvertToNumpyArray(maccs_bitvector,arr)
        # append the array to the empty list
        df_maccs.append(arr)

    MACCS = pd.concat([df, pd.DataFrame(df_maccs)], axis=1)
    # Remove single column
    MACCS = MACCS.drop(columns='smiles')
    MACCS = MACCS.drop(columns='mol')
    MACCS.to_csv('smiles_fp.csv',header=True,index = False)
    return MACCS
    
def classification():
    predict_x = pd.read_csv('smiles_fp.csv') 
    predict_res = {}
    predict_res['smiles'] = molecule
    targets =  ['AR','ER','GR','TR','PPARg','Aromatase'] #  ['ER'] #
    for target in targets:
    # enter the filename for specfic target
        filename1 = f'voting_model_{target}.pkl'
        # Open the file in binary read mode ('rb')
        with open(filename1, "rb") as file:
            loaded_model = pickle.load(file)
            y_pred = loaded_model.predict(predict_x)
            col_name = f'predict_{target}'
            predict_res[col_name] = y_pred
    df1 = pd.DataFrame(predict_res)
    return(df1)


# Upload SMILES string

molecule = st.text_input("Molecule", 'CC1CC2C3CC(F)C4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO')
molecule_list = [molecule]

if molecule is not None:
    # print (uploaded_file)
    # Save uploaded file to disk.
    smile_code = st_ketcher(molecule)
    st.markdown(f"Smile code: ``{smile_code}``")

    with st.form('predict_form'):
        submitted = st.form_submit_button('Predict')
        if submitted:
            with st.spinner('Wait for it...'):
                MACCS = from_smiles_to_csv(molecule_list)
                result = classification()
                print(result)
                output = ''
            with st.form('predict_form'):
                submitted = st.form_submit_button('AR')
                submitted = st.form_submit_button('ER')
                submitted = st.form_submit_button('GR')
                submitted = st.form_submit_button('TR')
                submitted = st.form_submit_button('PPARg')
                submitted = st.form_submit_button('Aromatase')
                    
                if result['predict_AR'] == 1:
                    output = f'AR_Yes'
                else:
                    output = f'AR_No'
                        
                if result['predict_ER'] == 1:
                    output = f'ER_Yes'
                else:
                    output = f'ER_No'
                        
                if result['predict_GR'] == 1:
                     output = f'GR_Yes'
                else:
                    output = f'GR_No'

                if result['predict_TR'] == 1:
                    output = f'TR_Yes'
                else:
                    output = f'TR_No'

                if result['predict_PPARg'] == 1:
                    output = f'PPARg_Yes'
                else:
                    output = f'PPARg_No'

                if result['predict_Aromatase'] == 1:
                    output = f'Aromatase_Yes'
                else:
                    output = f'Aromatase_No'
            st.success('Done!')
            st.markdown(f'<p style="font-size: 20px;"><b>{output}</b></p>', unsafe_allow_html=True)
            #st.balloons()


#Add a header and expander in side bar
original_title = '<p style="color:Blue; font-size: 20px;"><b>SMILES Prediction</b></p>'
st.sidebar.markdown(original_title, unsafe_allow_html=True)
with st.sidebar.expander("**About the App**"):
     st.markdown("""
        <span style='color:green'> Use this simple app to know whether the given chemical is a endocrine disruptor from this AI driven tool.
        Endocrine Disruptor contains various labels like 'AR','ER','GR','TR','PPARg','Aromatase', etc.
        This app was created for product demo. Hope you enjoy!</span>
     """, unsafe_allow_html=True)


#Add a feedback section in the sidebar
st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Please help us improve!')
with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
    text=st.text_input(label='Please leave your feedback here')
    submitted = st.form_submit_button('Submit')
    if submitted:
      st.write('Thanks for your feedback!')
      st.markdown('Your Rating:')
      st.markdown(rating)
      st.markdown('Your Feedback:')
      st.markdown(text)






