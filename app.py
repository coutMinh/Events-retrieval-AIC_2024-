import streamlit as st
from PIL import Image
from utils import MyFaiss
import os
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Query Tools")

######################################
bin_file = 'faiss_cosine.bin'
json_path = 'database/keyframes_id.json'
cosine_faiss = MyFaiss('', bin_file, json_path)
csv_file_path = 'temp.csv' 
######################################
st.sidebar.header("Query Inputs")
query = st.sidebar.text_input("Nhập query:")
id = st.sidebar.text_input("Nhập id query:")
neighbor = st.sidebar.text_input("Nhập id neighbor:")
######################################
query_button = st.sidebar.button('Search by Text Query')
id_button = st.sidebar.button('Search by ID Query')
neighbor_button = st.sidebar.button('Search neighbor')
save_ans = st.sidebar.button('Save csv')
######################################
num_cols = 4  # Number of columns for image display
cols = st.columns(num_cols)



def display_images(image_paths, ids, scores=None):
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        with cols[i % num_cols]:
            st.image(img, caption=f"{os.path.basename(image_path)} - id: {ids[i]}", use_column_width=True)

### Handle query button
if query and query_button:
  scores, id_queries, infos_query, image_paths = cosine_faiss.text_search(query, k=16)
  st.subheader("Results for Text Query")
  video_ans = [i.split('\\')[-2] for i in image_paths]
  keyframe_ans = [i.split('\\')[-1] for i in image_paths]
  df = pd.DataFrame({'video_ans': video_ans, 'keyframe_ans' : keyframe_ans})
  df.to_csv(csv_file_path, index=False)
  display_images(image_paths, id_queries, scores)


elif id and id_button:
  i_scores, id_queries, infos_query, image_paths = cosine_faiss.image_search(id_query=int(id), k=16)
  st.subheader("Results for ID Query")
  video_ans = [i.split('\\')[-2] for i in image_paths]
  keyframe_ans = [i.split('\\')[-1] for i in image_paths]
  df = pd.DataFrame({'video_ans': video_ans, 'keyframe_ans' : keyframe_ans})
  df.to_csv(csv_file_path, index=False)
  display_images(image_paths, id_queries, i_scores)

elif neighbor and neighbor_button:
   image_paths, id_queries = cosine_faiss.takeNeighbor(int(neighbor), k = 8)
   st.subheader("Results for neighbor")
   display_images(image_paths, id_queries)

elif save_ans and (id or query):
  df = pd.read_csv(csv_file_path)
  st.dataframe(df)
   
