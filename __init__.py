import streamlit as st 
import torch 
from torch import nn
from transformers import BertModel , AutoTokenizer
import re


st.set_page_config("Analisis de sentimientos", page_icon ="❤️")

st.title("Modelo de analisis de sentimientos ❤️")

model_name = "bert-base-multilingual-uncased"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BERTSentimentClassifier(nn.Module):
    def __init__(self,n_clases):
        super(BERTSentimentClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict = False) # se carga el modelo bert que es la primera parte de todo el modelo que se va a crear 
        self.drop = nn.Dropout(p=0.35) # durante el entrenamiento se apaga el 30% de las neuronas para hacerla mas robusta y que generalize mejor los datos
        self.linear = nn.Linear(self.bert.config.hidden_size, n_clases) # numero de neuronas de entrada a la capa lineal (las misma de la salida de bert 768), numero de neuronas de salida (los sentimientos a predecir 2)
    

    def forward (self,input_ids,attention_mask):
        _,cls_output = self.bert(  # bert arroja dos datos: _ contiene todos los embeddings de la codificacion de la clase de entrada y, cls_output que es el que interesa que es la codificacion del token [CLS] que es el que importa pues contiene "toda la escencia de la frase"
        input_ids = input_ids,
        attention_mask = attention_mask
        )
        drop_out = self.drop(cls_output)
        output = self.linear(drop_out)
        return output

    
tokenizer = AutoTokenizer.from_pretrained(model_name)

mod = st.sidebar.radio(
    "Escoja el modelos",
    ["Modelo 1","Modelo 2"],
    help = "El modelo 1 fue entrenado con un corpus balanceado. El modelo 2 fue entrenado en un corpus desbalanceado pero mayor"
    )

if mod =="Modelo 2":
    model = torch.load("/home/felipe/Escritorio/Personal/trabajo/modelos as/3labels_total/analisis_de_sentimiento_3labels_total.pth", map_location=device)
else:
    model = torch.load("/home/felipe/Escritorio/Personal/trabajo/modelos as/3labels_balanceado/analisis_de_sentimiento_3labels_balanceado.pth", map_location=device)
    

with st.form(key="my_form"):
    texto = st.text_area("Introduzca el texto", height=210)
    num_words = len(re.findall(r"\w+", texto))
    st.write('numero de palabras',num_words)
    submit_button = st.form_submit_button(label="✨ Obtener resultados")
    
    
max_words = 500
if num_words > max_words:
    st.write("El texto que introdujo es de mas de 500 palabras, se tomaran las primeras 500")
    texto = texto[:max_words]
    
if not submit_button:
    st.stop()
    

@st.cache(allow_output_mutation=True)
def classifySentiment (review_text):
    encoding_review = tokenizer.encode_plus( # se trae al codificador anteriormente creado para que codifique las entradas de manera correcta
                                                        review_text, 
                                                        max_length=150, 
                                                        truncation= True,   
                                                        add_special_tokens = True,   
                                                        return_token_type_ids= False,  
                                                        padding='longest', 
                                                        return_attention_mask= True,  
                                                        return_tensors=  'pt' 
  )
    
    input_ids = encoding_review['input_ids'].to(device)
    attention_mask = encoding_review['attention_mask'].to(device) 
    output = model(input_ids,attention_mask)
    soft = nn.Softmax(dim=1)
    porc = soft(output)
    _, prediccion = torch.max(output, dim=1)
    neg = porc[0][0].item()
    neu = porc[0][1].item()
    pos = porc[0][2].item()
    
    return neg, neu, pos, prediccion


neg, neu, pos, pred = classifySentiment(texto)
pred = pred.item()


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


