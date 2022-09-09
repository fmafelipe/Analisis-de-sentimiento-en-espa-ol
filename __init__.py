import streamlit as st 
import torch 
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re


st.set_page_config("Analisis de sentimientos", page_icon ="❤️")

st.title("Analisis de sentimientos EDS ❤️")


model_name = "SickBoy/analisis-sentimientos-spanish-eds"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
    

with st.form(key="my_form"):
    texto = st.text_area("Introduzca el texto", height=150)
    num_words = len(re.findall(r"\w+", texto))
    st.write('numero de palabras',num_words)
    submit_button = st.form_submit_button(label="✨ Obtener resultados")
    
max_words = 500
if num_words > max_words:
    st.write("El texto que introdujo es muy largo, este se ajustara a 512 tokens")
    texto = texto[:max_words]
    
if not submit_button:
    st.stop()
    

def classifySentiment (review_text):
    encode_text = tokenizer(review_text,
    padding = True,
    truncation = True,
    max_length = 512,
    return_tensors="pt")

    salida = model(**encode_text)
    predicciones = nn.functional.softmax(salida.logits, dim=1)
    neg = predicciones[0][0].item()
    neu = predicciones[0][1].item()
    pos = predicciones[0][2].item()
    mayor = torch.max(predicciones, dim=1)
    indice = mayor.indices.item()
    
    return neg, neu, pos, indice

neg, neu, pos, pred = classifySentiment(texto)



bar_neg = st.progress(neg)
st.write("negativo: ",neg)
bar_neu = st.progress(neu)
st.write("neutro: ",neu)
bar_pos = st.progress(pos)
st.write("positivo: ",pos)

if pred == 0:
    st.write("Sentimiento predicho: Negativo ")
elif pred == 1:
    st.write("Sentimiento predicho: Neutro ")
elif pred == 2:
    st.write("Sentimiento predicho: Positivo ")


