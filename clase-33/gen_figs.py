"""
Figuras clase 33 - Introduccion a NLP: la escalera de representacion.

HILO CONDUCTOR: "¿Como le ensenamos a una maquina a entender el SIGNIFICADO?"
Escalera de representacion, cada peldano forzado por el fracaso del anterior:
  0. El problema (mensajes ambiguos / phishing)
  1. Contar palabras (Bag-of-Words)            -> falla: "no me gusto" ≈ "me gusto"
  1.5 Pesar por rareza (Zipf, stopwords, TF-IDF) -> falla: fraseo nuevo se cuela
  2. Embeddings (significado = geometria, coseno) -> falla: vector fijo, sin contexto
  3. Atencion / Transformers (significado contextual)

Solo sklearn + gensim + matplotlib (NO tensorflow: TF+matplotlib deadlockea en macOS).
La figura de atencion (heatmap) es ILUSTRATIVA; el modelo Keras corre en el notebook.
"""
import os, warnings, re, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = ROOT
ARCA_RED, ARCA_DARK, ARCA_GRAY = "#C82B40", "#6B1525", "#F5F5F5"
ARCA_GREEN, ARCA_BLUE, ARCA_ORANGE, ARCA_PURPLE = "#16A34A", "#2563EB", "#EA580C", "#7C3AED"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "figure.dpi": 130})
np.random.seed(42)

SP_STOP = set('''el la los las un una unos unas de del a ante con en por para sin sobre y o u e
que se su sus lo le les es son ser como mas pero ya no si al me mi te tu esta este esto estas
estos esa ese eso esas esos muy ha han hay fue era porque cuando donde quien cual nos da va ni
tan todo toda todos todas solo uno tiene tienen hace hacer vez ver dos asi ademas aunque sino
hasta desde entre cada mientras tambien pelicula peliculas film films cine filme director
historia personajes personaje actor actores guion escena escenas'''.split())

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig); print(f"  guardada {name}")

def tokenize(t):
    t = t.lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-zñ]+", t)


# =============================================================================
#  DATOS + MODELOS (sklearn baselines + gensim Word2Vec) -> numeros reales
# =============================================================================
print("Cargando corpus...")
df = pd.read_csv(os.path.join(ROOT, "muchocine_sentimiento.csv")); df["review"] = df["review"].astype(str)
textos = np.array(df["review"].tolist()); labels = np.array(df["label"].astype(int).tolist())
spam = pd.read_csv(os.path.join(ROOT, "spam_es.csv")); spam["texto"] = spam["texto"].astype(str)
toks = [tokenize(t) for t in textos]
vocab_counter = Counter(w for d in toks for w in d)
print(f"  sentimiento {len(df)} | spam {len(spam)} | vocab {len(vocab_counter)}")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Baseline sentimiento
Xtr, Xte, ytr, yte = train_test_split(textos, labels, test_size=0.2, random_state=42, stratify=labels)
vec_s = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=3, ngram_range=(1,2), max_features=30000)
clf_s = LogisticRegression(max_iter=1000, C=3.0).fit(vec_s.fit_transform(Xtr), ytr)
pred_s = clf_s.predict(vec_s.transform(Xte))
acc_sent = accuracy_score(yte, pred_s); cm_sent = confusion_matrix(yte, pred_s)
print(f"  sentimiento acc = {acc_sent:.3f}")

# Baseline spam
Xs = np.array(spam["texto"].tolist()); ys = np.array(spam["label"].astype(int).tolist())
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.25, random_state=42, stratify=ys)
vec_sp = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), min_df=2)
clf_sp = LogisticRegression(max_iter=1000, C=5).fit(vec_sp.fit_transform(Xstr), ystr)
acc_spam = accuracy_score(yste, clf_sp.predict(vec_sp.transform(Xste)))
print(f"  spam acc = {acc_spam:.3f}")
EVAS = [("Su paquete esta retenido en aduana, abone la tasa para liberarlo", 1),
        ("Hemos bloqueado su tarjeta por seguridad, reactivela aqui", 1),
        ("Oye llegas a la cena de manana? confirma porfa", 0)]
evas_res = [(m, real, clf_sp.predict_proba(vec_sp.transform([m]))[0,1]) for m, real in EVAS]

# Word2Vec (gensim) sobre las resenas -> embeddings reales
print("Entrenando Word2Vec...")
from gensim.models import Word2Vec
w2v = Word2Vec(toks, vector_size=100, window=5, min_count=5, epochs=20, workers=1, seed=42, sg=1)
print(f"  vocab Word2Vec = {len(w2v.wv)}")


# =============================================================================
#  BLOQUE 0 - PROBLEMA + MAPA
# =============================================================================
def fig_hero():
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.add_patch(Rectangle((0,0),12,6,facecolor=ARCA_DARK))
    for w,x,c in [("texto",1.0,ARCA_GRAY),("→",2.9,ARCA_RED),("significado",3.6,"white"),
                  ("→",7.4,ARCA_RED),("decisión",8.1,ARCA_GREEN)]:
        ax.text(x,4.6,w,fontsize=25,color=c,fontweight="bold",va="center")
    ax.text(0.9,2.7,"Introducción a NLP",fontsize=34,color="white",fontweight="bold")
    ax.text(0.95,1.5,"La escalera de la representación: de contar palabras a entender el contexto",
            fontsize=14,color="#E5B5BD")
    save(fig,"fig_hero.png")

def fig_problema_descubrimiento():
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off"); ax.grid(False)
    ax.text(6,6.6,"Eres la máquina. Decide:",ha="center",fontsize=14,fontweight="bold",color=ARCA_DARK)
    msgs = [
        ('"La fotografía es preciosa, pero la trama me aburrió"', "¿positivo o negativo?", ARCA_ORANGE),
        ('"No me gustó nada, esperaba mucho más"', "¿positivo o negativo?", ARCA_BLUE),
        ('"Su cuenta será suspendida. Verifique sus datos aquí"', "¿spam o legítimo?", ARCA_RED),
        ('"Oye, ¿llegas a la cena? confirma porfa"', "¿spam o legítimo?", ARCA_GREEN),
    ]
    for i,(m,q,c) in enumerate(msgs):
        y = 5.4 - i*1.25
        ax.add_patch(FancyBboxPatch((0.5,y),8.4,1.0,boxstyle="round,pad=0.03,rounding_size=0.1",
                     facecolor=c,alpha=0.10,edgecolor=c,lw=2))
        ax.text(0.8,y+0.5,m,fontsize=11,va="center",color="#2D2D2D",style="italic")
        ax.text(9.2,y+0.5,q,fontsize=10,va="center",color=c,fontweight="bold")
    ax.text(6,0.35,"Para ti es trivial. Pero la máquina solo ve una secuencia de caracteres.\n"
            "Toda la clase es: ¿cómo le damos el SIGNIFICADO?",ha="center",fontsize=10.5,
            color="#6B7280")
    save(fig,"fig_problema_descubrimiento.png")

def _escalera(ax, highlight=None, done=None):
    done = done or set()
    pasos = [("1. Contar palabras","Bag-of-Words",ARCA_BLUE),
             ("2. Pesar por rareza","TF-IDF",ARCA_ORANGE),
             ("3. Significado","Embeddings + coseno",ARCA_GREEN),
             ("4. Contexto","Atención / Transformers",ARCA_PURPLE)]
    for i,(t,s,c) in enumerate(pasos):
        x, y = 0.5+i*2.95, 0.6+i*1.15
        on = (highlight==i); fin = (i in done)
        ax.add_patch(FancyBboxPatch((x,y),2.7,1.5,boxstyle="round,pad=0.04,rounding_size=0.12",
                     facecolor=c,alpha=0.30 if on else (0.16 if fin else 0.08),
                     edgecolor=c,lw=3 if on else 1.6))
        ax.text(x+1.35,y+1.0,t,ha="center",fontsize=10.5,fontweight="bold",color=c)
        ax.text(x+1.35,y+0.45,s,ha="center",fontsize=9,color="#2D2D2D")
        if fin: ax.text(x+2.45,y+1.25,"✓",ha="center",fontsize=13,color=c,fontweight="bold")
        if i<3:
            ax.add_patch(FancyArrowPatch((x+2.7,y+0.75),(x+2.95,y+1.9),arrowstyle="-|>",
                         mutation_scale=14,color="#999",lw=1.5))

def fig_escalera():
    fig, ax = plt.subplots(figsize=(12.5,5)); ax.set_xlim(0,12.5); ax.set_ylim(0,6.5); ax.axis("off"); ax.grid(False)
    ax.text(6.25,6.1,"La escalera de la representación: cada peldaño entiende más SIGNIFICADO",
            ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    _escalera(ax)
    ax.text(6.25,0.1,"Cada peldaño existe porque el anterior fracasó. Hoy los subimos todos.",
            ha="center",fontsize=10,color="#6B7280",style="italic")
    save(fig,"fig_escalera.png")

def fig_escalera_completa():
    fig, ax = plt.subplots(figsize=(12.5,5.2)); ax.set_xlim(0,12.5); ax.set_ylim(0,6.5); ax.axis("off"); ax.grid(False)
    ax.text(6.25,6.2,"Lo que subimos hoy",ha="center",fontsize=14,fontweight="bold",color=ARCA_DARK)
    _escalera(ax, done={0,1,2,3})
    notas = ["sentiment 84%\nspam 93%","corrige a Zipf","vecinos +\nbúsqueda coseno","la base\nde los LLMs"]
    for i,n in enumerate(notas):
        ax.text(0.5+i*2.95+1.35, 0.2, n, ha="center", fontsize=8, color="#6B7280")
    save(fig,"fig_escalera_completa.png")

def fig_que_es_nlp():
    fig, ax = plt.subplots(figsize=(12,6.2)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off"); ax.grid(False)
    cuad=[(0.3,5.2,ARCA_RED,"Clasificación",["Sentimiento","Detección de spam","Ruteo de tickets","Tema del documento"]),
          (5.1,5.2,ARCA_BLUE,"Extracción (NER)",["Marcas y productos","Lugares, fechas","Montos en facturas","Entidades en contratos"]),
          (0.3,0.3,ARCA_GREEN,"Similitud / Búsqueda",["FAQ y soporte","Duplicados","Recomendación","Búsqueda semántica"]),
          (5.1,0.3,ARCA_PURPLE,"Generación",["Resúmenes","Traducción","Chatbots","Respuestas auto."])]
    for x,y,col,tit,items in cuad:
        ax.add_patch(FancyBboxPatch((x,y),4.55,4.4,boxstyle="round,pad=0.08,rounding_size=0.2",
                     facecolor=col,alpha=0.10,edgecolor=col,lw=2.2))
        ax.text(x+0.3,y+3.95,tit,fontsize=14,fontweight="bold",color=col)
        for i,it in enumerate(items): ax.text(x+0.45,y+3.25-i*0.72,f"• {it}",fontsize=10.5,color="#2D2D2D")
    ax.text(5.0,9.75,"Las 4 familias de problemas de NLP (hoy: clasificación y similitud)",
            ha="center",fontsize=13,fontweight="bold",color=ARCA_DARK)
    save(fig,"fig_que_es_nlp.png")


# =============================================================================
#  BLOQUE 1 - TEXTO A NUMEROS
# =============================================================================
def fig_texto_a_numeros():
    fig, ax = plt.subplots(figsize=(12.5,4.8)); ax.set_xlim(0,13); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    et=[("TEXTO",'"No me gustó"',ARCA_DARK),("CARACTERES","N o  m e ...",ARCA_PURPLE),
        ("TOKENS","[no][me][gusto]",ARCA_BLUE),("VOCABULARIO","no→7 me→3\ngusto→42",ARCA_ORANGE),
        ("VECTOR","[7, 3, 42]",ARCA_GREEN)]
    n=len(et); w=2.05; gap=(13-n*w)/(n+1)
    for i,(t,e,c) in enumerate(et):
        x=gap+i*(w+gap)
        ax.add_patch(FancyBboxPatch((x,2.0),w,2.0,boxstyle="round,pad=0.06,rounding_size=0.15",
                     facecolor=c,alpha=0.12,edgecolor=c,lw=2))
        ax.text(x+w/2,3.55,t,ha="center",fontsize=10.5,fontweight="bold",color=c)
        ax.text(x+w/2,2.7,e,ha="center",fontsize=9.5,color="#2D2D2D",family="monospace")
        if i<n-1: ax.add_patch(FancyArrowPatch((x+w,3.0),(x+w+gap,3.0),arrowstyle="-|>",mutation_scale=18,color=ARCA_RED,lw=2))
    ax.text(6.5,5.3,"El texto se vuelve números: la decisión es DÓNDE cortar",ha="center",fontsize=13,fontweight="bold",color=ARCA_DARK)
    ax.text(6.5,1.2,"Igual que una imagen es una matriz de píxeles y una serie una secuencia de valores,\nun texto es ahora una secuencia de números.",ha="center",fontsize=10,color="#6B7280")
    save(fig,"fig_texto_a_numeros.png")

def fig_caracter_unicode():
    fig, axes = plt.subplots(1,2,figsize=(12.5,4.3))
    for ax in axes: ax.axis("off"); ax.grid(False)
    ax=axes[0]; ax.text(0.5,0.92,"Un carácter es un número (Unicode)",fontsize=12,fontweight="bold",color=ARCA_DARK,ha="center",transform=ax.transAxes)
    for i,c in enumerate("café"):
        x=0.12+i*0.22
        ax.text(x,0.6,c,fontsize=30,ha="center",color=ARCA_BLUE,transform=ax.transAxes)
        ax.text(x,0.4,f"ord={ord(c)}",fontsize=10,ha="center",color="#2D2D2D",transform=ax.transAxes,family="monospace")
    ax.text(0.5,0.18,"'ñ' = 241    ' ' = 32    '!' = 33",fontsize=10.5,ha="center",color="#6B7280",transform=ax.transAxes,family="monospace")
    ax=axes[1]; ax.text(0.5,0.92,"El mismo texto, distinto nº de caracteres",fontsize=12,fontweight="bold",color=ARCA_DARK,ha="center",transform=ax.transAxes)
    ax.text(0.5,0.60,"'café' (NFC): 4 caracteres\né = un code point (233)",fontsize=11.5,ha="center",color=ARCA_GREEN,transform=ax.transAxes,family="monospace")
    ax.text(0.5,0.30,"'café' (NFD): 5 caracteres\ne + ´ combinado (101+769)",fontsize=11.5,ha="center",color=ARCA_RED,transform=ax.transAxes,family="monospace")
    ax.text(0.5,0.05,"Por eso el texto se NORMALIZA antes de modelar.",fontsize=10,ha="center",color="#6B7280",style="italic",transform=ax.transAxes)
    fig.tight_layout(); save(fig,"fig_caracter_unicode.png")


# =============================================================================
#  BLOQUE 2 - BoW + SENTIMIENTO
# =============================================================================
def fig_corpus_bow():
    fig, ax = plt.subplots(figsize=(12.5,5.2)); ax.set_xlim(0,13); ax.set_ylim(0,7); ax.axis("off"); ax.grid(False)
    ax.text(6.5,6.6,"Corpus → Bag-of-Words: cuento palabras y tiro el orden",ha="center",fontsize=13.5,fontweight="bold",color=ARCA_DARK)
    for i,d in enumerate(['D1: "gratis gana premio gratis"','D2: "hola la reunión es hoy"']):
        ax.add_patch(FancyBboxPatch((0.4,4.7-i*1.0),4.7,0.8,boxstyle="round,pad=0.03,rounding_size=0.08",facecolor=ARCA_GRAY,edgecolor=ARCA_DARK,lw=1.2))
        ax.text(0.6,5.1-i*1.0,d,fontsize=10.5,va="center",color="#2D2D2D")
    ax.text(2.7,5.9,"CORPUS = colección de documentos",ha="center",fontsize=10,color=ARCA_PURPLE,fontweight="bold")
    ax.add_patch(FancyArrowPatch((5.3,4.5),(6.4,4.5),arrowstyle="-|>",mutation_scale=18,color=ARCA_RED,lw=2))
    vocab=["gratis","gana","premio","hola","reunion","hoy"]; M=[[2,1,1,0,0,0],[0,0,0,1,1,1]]
    x0,y0,cw,ch=6.7,3.2,0.95,0.7
    for j,v in enumerate(vocab): ax.text(x0+j*cw+cw/2,y0+2.1,v,ha="center",rotation=35,fontsize=8.5,color=ARCA_DARK)
    for i in range(2):
        ax.text(x0-0.15,y0+1.4-i*ch+ch/2,f"D{i+1}",ha="right",va="center",fontsize=9,color=ARCA_DARK)
        for j in range(6):
            val=M[i][j]
            ax.add_patch(Rectangle((x0+j*cw,y0+1.4-i*ch),cw,ch,facecolor=ARCA_BLUE,alpha=0.12+0.30*val,edgecolor="white",lw=1.5))
            ax.text(x0+j*cw+cw/2,y0+1.4-i*ch+ch/2,str(val),ha="center",va="center",fontsize=11,color="#2D2D2D",fontweight="bold")
    ax.text(9.5,1.5,"matriz documento × palabra\n(cada fila = un vector)",ha="center",fontsize=9.5,color="#6B7280")
    ax.text(6.5,0.5,'"el perro muerde al hombre" y "el hombre muerde al perro" → MISMA bolsa. El orden se pierde.',
            ha="center",fontsize=10,color=ARCA_ORANGE,style="italic")
    save(fig,"fig_corpus_bow.png")

def fig_eda_ngramas():
    def top_c(mask,n=12):
        c=Counter(w for d,m in zip(toks,mask) if m for w in d if w not in SP_STOP and len(w)>2)
        return c.most_common(n)
    pos=top_c(labels==1)[::-1]; neg=top_c(labels==0)[::-1]
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    axes[0].barh([w for w,_ in pos],[v for _,v in pos],color=ARCA_GREEN,alpha=0.8); axes[0].set_title("Top palabras — POSITIVAS")
    axes[1].barh([w for w,_ in neg],[v for _,v in neg],color=ARCA_RED,alpha=0.8); axes[1].set_title("Top palabras — NEGATIVAS")
    for ax in axes: ax.set_xlabel("frecuencia"); ax.grid(True,axis="x",alpha=0.2)
    fig.tight_layout(); save(fig,"fig_eda_ngramas.png")

def fig_sentiment_confusion():
    fig, ax = plt.subplots(figsize=(5.6,5)); ax.imshow(cm_sent,cmap="Blues"); ax.grid(False)
    ax.set_xticks([0,1]); ax.set_xticklabels(["negativo","positivo"]); ax.set_xlabel("predicho")
    ax.set_yticks([0,1]); ax.set_yticklabels(["negativo","positivo"]); ax.set_ylabel("real")
    ax.set_title(f"Sentimiento — Regresión Logística\nacc = {acc_sent:.1%}")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm_sent[i,j],ha="center",va="center",fontsize=18,color="white" if cm_sent[i,j]>cm_sent.max()*0.6 else "#2D2D2D")
    fig.tight_layout(); save(fig,"fig_sentiment_confusion.png")

def fig_sentiment_coefs():
    fn=np.array(vec_s.get_feature_names_out()); co=clf_s.coef_[0]
    tp=np.argsort(co)[-12:]; tn=np.argsort(co)[:12]
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    axes[0].barh(fn[tp],co[tp],color=ARCA_GREEN,alpha=0.85); axes[0].set_title("Empujan a POSITIVO")
    axes[1].barh(fn[tn],co[tn],color=ARCA_RED,alpha=0.85); axes[1].set_title("Empujan a NEGATIVO")
    for ax in axes: ax.axvline(0,color="black",lw=0.6); ax.set_xlabel("peso en la regresión")
    fig.tight_layout(); save(fig,"fig_sentiment_coefs.png")

def fig_bow_falla():
    from sklearn.feature_extraction.text import CountVectorizer
    a,b="me gusto la pelicula","no me gusto la pelicula"
    M=CountVectorizer().fit_transform([a,b]).toarray(); sim=cosine_similarity(M)[0,1]
    fig, ax = plt.subplots(figsize=(11,4.4)); ax.axis("off"); ax.grid(False)
    ax.text(0.5,0.95,"Fracaso del peldaño 1: el orden y la negación se pierden",fontsize=13.5,fontweight="bold",color=ARCA_DARK,ha="center",transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.06,0.55),0.40,0.28,boxstyle="round,pad=0.02,rounding_size=0.04",facecolor=ARCA_GREEN,alpha=0.12,edgecolor=ARCA_GREEN,lw=2,transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.54,0.55),0.40,0.28,boxstyle="round,pad=0.02,rounding_size=0.04",facecolor=ARCA_RED,alpha=0.12,edgecolor=ARCA_RED,lw=2,transform=ax.transAxes))
    ax.text(0.26,0.69,f'"{a}"\nPOSITIVO',ha="center",va="center",fontsize=11.5,color=ARCA_GREEN,fontweight="bold",transform=ax.transAxes)
    ax.text(0.74,0.69,f'"{b}"\nNEGATIVO',ha="center",va="center",fontsize=11.5,color=ARCA_RED,fontweight="bold",transform=ax.transAxes)
    ax.text(0.5,0.34,f"Similitud coseno de sus vectores BoW = {sim:.2f}",ha="center",fontsize=14,fontweight="bold",color=ARCA_ORANGE,transform=ax.transAxes)
    ax.text(0.5,0.16,"Significan lo OPUESTO, pero para el BoW son casi idénticos: solo cambia 'no'.",ha="center",fontsize=10.5,color="#6B7280",transform=ax.transAxes)
    save(fig,"fig_bow_falla.png")


# =============================================================================
#  BLOQUE 3 - ZIPF / STOPWORDS / TF-IDF / SPAM
# =============================================================================
def fig_zipf():
    fig, ax = plt.subplots(figsize=(10.5,5))
    counts=np.array(sorted(vocab_counter.values(),reverse=True)); ranks=np.arange(1,len(counts)+1)
    ax.loglog(ranks,counts,color=ARCA_BLUE,lw=1.8,label="corpus real")
    ax.loglog(ranks,counts[0]/ranks,color=ARCA_RED,ls="--",lw=1.4,label="Zipf ideal (1/rango)")
    for i,w in enumerate([w for w,_ in vocab_counter.most_common(4)]):
        ax.annotate(w,(i+1,counts[i]),fontsize=10,color=ARCA_DARK,xytext=(6,4),textcoords="offset points")
    ax.annotate("la PUNTA: stopwords\n(en todos los docs,\nno discriminan)",xy=(2,counts[1]),xytext=(6,counts[0]*0.5),fontsize=9,color=ARCA_DARK,arrowprops=dict(arrowstyle="->",color=ARCA_DARK,lw=1))
    ax.annotate("la COLA larga: palabras raras\n(matriz enorme y dispersa;\nsiempre habrá palabras nuevas)",xy=(len(counts)*0.5,counts[int(len(counts)*0.5)]),xytext=(40,counts[0]*0.02),fontsize=9,color=ARCA_ORANGE,arrowprops=dict(arrowstyle="->",color=ARCA_ORANGE,lw=1))
    ax.set_title("Ley de Zipf: pocas palabras dominan, una cola larguísima de palabras raras")
    ax.set_xlabel("rango (log)"); ax.set_ylabel("frecuencia (log)"); ax.legend(); ax.grid(True,which="both",alpha=0.2)
    fig.tight_layout(); save(fig,"fig_zipf.png")

def fig_stopwords():
    fig, ax = plt.subplots(figsize=(11.5,4.8)); ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.text(5.5,5.6,"Stopwords: las palabras más frecuentes... y menos informativas",ha="center",fontsize=13,fontweight="bold",color=ARCA_DARK)
    top=[w for w,_ in vocab_counter.most_common(12)]
    ax.add_patch(FancyBboxPatch((0.4,3.3),10.2,1.4,boxstyle="round,pad=0.04,rounding_size=0.1",facecolor=ARCA_BLUE,alpha=0.10,edgecolor=ARCA_BLUE,lw=2))
    ax.text(0.6,4.35,"Top del corpus:",fontsize=10,color=ARCA_BLUE,fontweight="bold")
    ax.text(0.6,3.75,"  ".join(top),fontsize=12,color="#2D2D2D",family="monospace")
    ax.text(5.5,2.7,"Están en TODOS los documentos → no distinguen de qué trata ninguno.",ha="center",fontsize=10.5,color="#6B7280")
    ax.add_patch(FancyBboxPatch((0.4,0.3),10.2,1.9,boxstyle="round,pad=0.04,rounding_size=0.1",facecolor=ARCA_RED,alpha=0.08,edgecolor=ARCA_RED,lw=2))
    ax.text(5.5,1.85,"El matiz honesto: quitarlas NO es un dogma",ha="center",fontsize=11,fontweight="bold",color=ARCA_RED)
    ax.text(5.5,1.25,'"No me gustó" → "no" es stopword pero INVIERTE el significado.',ha="center",fontsize=10,color="#2D2D2D")
    ax.text(5.5,0.7,"Dependen del idioma y del dominio; los modelos modernos NO las quitan.",ha="center",fontsize=9.5,color="#6B7280")
    save(fig,"fig_stopwords.png")

def fig_bow_tfidf():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer as TV
    docs=["la pelicula es buena","la pelicula es muy buena","la pelicula es mala"]
    cv=CountVectorizer(); bow=cv.fit_transform(docs).toarray()
    tv=TV(); tfidf=tv.fit_transform(docs).toarray(); vocab=cv.get_feature_names_out()
    fig, axes = plt.subplots(1,2,figsize=(13,4.2))
    for ax,M,tit,cmap,isf in [(axes[0],bow,"Bag-of-Words (conteo)","Blues",False),(axes[1],tfidf,"TF-IDF (peso por rareza)","Oranges",True)]:
        ax.imshow(M,cmap=cmap,aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab,rotation=45,ha="right",fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels([f"doc {i+1}" for i in range(3)],fontsize=9); ax.set_title(tit)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j,i,f"{M[i,j]:.2f}" if isf else f"{int(M[i,j])}",ha="center",va="center",fontsize=8,color="white" if M[i,j]>M.max()*0.6 else "#2D2D2D")
    fig.tight_layout(); save(fig,"fig_bow_tfidf.png")

def fig_spam_resultado():
    fig, axes = plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={"width_ratios":[1,1.5]})
    ax=axes[0]; ax.bar(["LogReg\nspam"],[acc_spam*100],color=ARCA_GREEN,alpha=0.85,width=0.45)
    ax.text(0,acc_spam*100+0.5,f"{acc_spam*100:.1f}%",ha="center",fontsize=15,fontweight="bold")
    ax.axhline(50,color="gray",ls=":",lw=1); ax.set_ylim(45,100); ax.set_ylabel("accuracy en test (%)"); ax.set_title("El clásico GANA en spam limpio")
    ax=axes[1]; ax.axis("off"); ax.grid(False)
    ax.text(0.5,0.95,"...pero es FRÁGIL con fraseo nuevo",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_RED,transform=ax.transAxes)
    y=0.72
    for m,real,p in evas_res:
        pred=int(p>=0.5); ok=pred==real; col=ARCA_GREEN if ok else ARCA_RED
        ax.text(0.02,y,("✗ " if not ok else "✓ ")+f"p(spam)={p:.2f} → {'spam' if pred else 'ham'}",fontsize=10,color=col,fontweight="bold",transform=ax.transAxes)
        ax.text(0.02,y-0.07,f'   "{m[:52]}..."  (real: {"spam" if real else "ham"})',fontsize=8.5,color="#6B7280",transform=ax.transAxes)
        y-=0.22
    ax.text(0.5,0.02,"Deja pasar phishing y marca como spam un mensaje normal.",ha="center",fontsize=9.5,color="#6B7280",style="italic",transform=ax.transAxes)
    fig.tight_layout(); save(fig,"fig_spam_resultado.png")


# =============================================================================
#  BLOQUE 4 - EMBEDDINGS (REALES con Word2Vec) + COSENO
# =============================================================================
def fig_embeddings_concepto():
    fig, ax = plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.text(6,5.6,"Embeddings: cada palabra es un VECTOR; el significado es geometría",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    # mini plano 2D con palabras agrupadas
    pts={"excelente":(2.0,3.8,ARCA_GREEN),"buena":(2.5,3.2,ARCA_GREEN),"genial":(1.6,3.3,ARCA_GREEN),
         "mala":(8.2,1.4,ARCA_RED),"aburrida":(8.8,1.9,ARCA_RED),"pésima":(8.3,2.2,ARCA_RED)}
    for w,(x,y,c) in pts.items():
        ax.add_patch(FancyArrowPatch((6,2.6),(x,y),arrowstyle="-|>",mutation_scale=10,color=c,lw=1.2,alpha=0.5))
        ax.text(x,y+0.12,w,ha="center",fontsize=11,color=c,fontweight="bold")
    ax.add_patch(Circle((6,2.6),0.06,color="#333"))
    ax.add_patch(FancyBboxPatch((1.0,2.9),2.2,1.5,boxstyle="round,pad=0.05",facecolor=ARCA_GREEN,alpha=0.06,edgecolor=ARCA_GREEN,lw=1.5,ls="--"))
    ax.add_patch(FancyBboxPatch((7.6,1.0),2.0,1.6,boxstyle="round,pad=0.05",facecolor=ARCA_RED,alpha=0.06,edgecolor=ARCA_RED,lw=1.5,ls="--"))
    ax.text(6,0.5,"Palabras de significado parecido quedan CERCA. Conecta con PCA y el espacio latente del autoencoder (clase 29).",ha="center",fontsize=10,color="#6B7280")
    save(fig,"fig_embeddings_concepto.png")

def fig_word2vec_idea():
    fig, ax = plt.subplots(figsize=(12,4.6)); ax.set_xlim(0,12); ax.set_ylim(0,5.5); ax.axis("off"); ax.grid(False)
    ax.text(6,5.1,"¿Cómo aprende Word2Vec? Por las compañías que tiene una palabra",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    frase=["la","trama","era","___","y","me","aburrió"]
    for i,w in enumerate(frase):
        x=1.2+i*1.4; tgt=(w=="___")
        ax.add_patch(FancyBboxPatch((x,2.6),1.2,0.9,boxstyle="round,pad=0.04,rounding_size=0.1",facecolor=ARCA_ORANGE if tgt else ARCA_BLUE,alpha=0.18 if tgt else 0.10,edgecolor=ARCA_ORANGE if tgt else ARCA_BLUE,lw=2))
        ax.text(x+0.6,3.05,w,ha="center",va="center",fontsize=11,color="#2D2D2D")
    ax.text(6,1.9,'El modelo predice la palabra oculta a partir de su CONTEXTO ("la trama era ___ y me aburrió" → "lenta", "mala").',ha="center",fontsize=10,color="#2D2D2D")
    ax.text(6,1.1,"Resultado: palabras que aparecen en contextos parecidos terminan con vectores parecidos.",ha="center",fontsize=10.5,color=ARCA_GREEN,fontweight="bold")
    ax.text(6,0.4,"(No le decimos qué significan: lo deduce de millones de frases. Aprendizaje auto-supervisado.)",ha="center",fontsize=9,color="#6B7280",style="italic")
    save(fig,"fig_word2vec_idea.png")

def fig_embeddings_vecinos():
    palabras=["buena","aburrida","excelente","actores","obra"]
    palabras=[w for w in palabras if w in w2v.wv]
    fig, axes = plt.subplots(1,len(palabras),figsize=(13,4.2))
    for ax,w in zip(axes,palabras):
        nn=w2v.wv.most_similar(w,topn=6)[::-1]
        ax.barh([x for x,_ in nn],[s for _,s in nn],color=ARCA_PURPLE,alpha=0.8)
        ax.set_title(f'"{w}"',fontsize=12,color=ARCA_DARK); ax.set_xlim(0,1); ax.tick_params(labelsize=9)
        ax.grid(True,axis="x",alpha=0.2)
    fig.suptitle("Vecinos por significado (Word2Vec entrenado en las reseñas) — similitud coseno",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    fig.tight_layout(); save(fig,"fig_embeddings_vecinos.png")

def fig_embeddings_2d():
    seed_pos=["buena","excelente","gran","mejor","maravillosa","brillante","obra","estupenda","entretenida","genial"]
    seed_neg=["mala","aburrida","tediosa","lenta","pesima","horrible","peor","floja","monotona","ridicula"]
    base=[w for w,_ in vocab_counter.most_common(400) if w in w2v.wv and w not in SP_STOP and len(w)>3][:120]
    words=list(dict.fromkeys(base+[w for w in seed_pos+seed_neg if w in w2v.wv]))
    V=np.array([w2v.wv[w] for w in words]); P=PCA(n_components=2,random_state=42).fit_transform(V)
    fig, ax = plt.subplots(figsize=(11,6.5))
    for (x,y),w in zip(P,words):
        if w in seed_pos: ax.scatter(x,y,c=ARCA_GREEN,s=45,zorder=3); ax.text(x,y+0.03,w,fontsize=9,color=ARCA_GREEN,fontweight="bold",ha="center")
        elif w in seed_neg: ax.scatter(x,y,c=ARCA_RED,s=45,zorder=3); ax.text(x,y+0.03,w,fontsize=9,color=ARCA_RED,fontweight="bold",ha="center")
        else: ax.scatter(x,y,c="#CBD5E1",s=10,zorder=1)
    ax.set_title("Mapa 2D de los embeddings (PCA): los sinónimos se agrupan\n(p. ej. 'lenta/tediosa/monótona/aburrida' caen juntas a la derecha)")
    ax.set_xlabel("componente 1"); ax.set_ylabel("componente 2"); ax.grid(True,alpha=0.2)
    fig.tight_layout(); save(fig,"fig_embeddings_2d.png")

def fig_coseno_concepto():
    fig, ax = plt.subplots(figsize=(11,4.6)); ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.text(5.5,5.6,"Similitud coseno: el ÁNGULO entre dos vectores mide cuánto se parecen",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    import numpy as _np
    ox,oy=2.6,1.0
    pairs=[((2.2,3.4),ARCA_GREEN,"comentario A"),((3.0,3.0),ARCA_GREEN,"comentario B (parecido)"),((0.2,3.0),ARCA_RED,"comentario C (distinto)")]
    for (dx,dy),c,lab in pairs:
        ax.add_patch(FancyArrowPatch((ox,oy),(ox+dx,oy+dy),arrowstyle="-|>",mutation_scale=16,color=c,lw=2.2))
        ax.text(ox+dx,oy+dy+0.15,lab,fontsize=9,color=c,ha="center")
    ax.text(7.6,3.6,"cos(θ) = 1   → idénticos",fontsize=11,color=ARCA_GREEN,fontweight="bold")
    ax.text(7.6,3.0,"cos(θ) = 0   → sin relación",fontsize=11,color="#6B7280")
    ax.text(7.6,2.4,"cos(θ) = -1  → opuestos",fontsize=11,color=ARCA_RED)
    ax.text(5.5,0.4,"Comparamos por DIRECCIÓN, no por longitud: dos textos del mismo tema apuntan al mismo lado.",ha="center",fontsize=10,color="#6B7280")
    save(fig,"fig_coseno_concepto.png")

def fig_busqueda_coseno():
    # vector de documento = promedio de vectores de palabra; busqueda por coseno
    def docvec(t):
        vs=[w2v.wv[w] for w in tokenize(t) if w in w2v.wv]
        return np.mean(vs,axis=0) if vs else np.zeros(w2v.vector_size)
    D=np.array([docvec(t) for t in textos])
    query="una de las mejores peliculas que he visto, una obra maestra"
    q=docvec(query); sims=cosine_similarity([q],D)[0]; top=np.argsort(-sims)[:3]
    fig, ax = plt.subplots(figsize=(12,4.8)); ax.axis("off"); ax.grid(False)
    ax.text(0.5,0.95,"Búsqueda por SIGNIFICADO (coseno sobre embeddings), no por palabras",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK,transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.04,0.74),0.92,0.13,boxstyle="round,pad=0.01,rounding_size=0.02",facecolor=ARCA_PURPLE,alpha=0.10,edgecolor=ARCA_PURPLE,lw=2,transform=ax.transAxes))
    ax.text(0.06,0.805,f'Consulta:  "{query}"',fontsize=10.5,color=ARCA_DARK,va="center",transform=ax.transAxes)
    y=0.60
    for rank,i in enumerate(top,1):
        s=sims[i]; txt=re.sub(r"\s+"," ",textos[i])[:90]
        ax.text(0.06,y,f"#{rank}  coseno={s:.2f}",fontsize=10,color=ARCA_GREEN,fontweight="bold",transform=ax.transAxes)
        ax.text(0.06,y-0.06,f'   "{txt}..."',fontsize=9,color="#2D2D2D",transform=ax.transAxes)
        y-=0.17
    ax.text(0.5,0.02,"Recupera reseñas con el mismo sentido aunque usen otras palabras. Es la base del buscador semántico y del RAG (Módulo 6).",ha="center",fontsize=9,color="#6B7280",style="italic",transform=ax.transAxes)
    save(fig,"fig_busqueda_coseno.png")


# =============================================================================
#  BLOQUE 5 - ATENCION / TRANSFORMERS
# =============================================================================
def fig_atencion_concepto():
    fig, ax = plt.subplots(figsize=(12,5)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.text(6,5.6,"Atención: cada palabra mira a TODAS y pondera cuáles le importan",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    palabras=["no","me","gustó","nada"]; xs=[2.0,4.0,6.0,8.0]
    for x,w in zip(xs,palabras):
        ax.add_patch(Circle((x,3.2),0.55,facecolor=ARCA_BLUE,alpha=0.12,edgecolor=ARCA_BLUE,lw=2))
        ax.text(x,3.2,w,ha="center",va="center",fontsize=12,color="#2D2D2D")
    for x,peso,c in [(2.0,0.9,ARCA_RED),(4.0,0.2,"#BBB"),(8.0,0.6,ARCA_ORANGE)]:
        ax.add_patch(FancyArrowPatch((6.0,3.75),(x,3.75),connectionstyle="arc3,rad=0.35",arrowstyle="-|>",mutation_scale=14,color=c,lw=1+3*peso,alpha=0.4+0.5*peso))
    ax.text(6,1.8,'"gustó" aprende a mirar fuerte a "no" → entiende la NEGACIÓN',ha="center",fontsize=11,color=ARCA_RED,fontweight="bold")
    ax.text(6,1.1,"El peso de cada flecha se APRENDE de los datos. Eso es lo que el Bag-of-Words no podía hacer.",ha="center",fontsize=10,color="#6B7280")
    save(fig,"fig_atencion_concepto.png")

def fig_qkv():
    fig, ax = plt.subplots(figsize=(12.5,5.4)); ax.set_xlim(0,12.5); ax.set_ylim(0,6.5); ax.axis("off"); ax.grid(False)
    ax.text(6.25,6.1,"Por dentro: Query, Key, Value (cómo se calcula el peso)",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    steps=[("1. Query","La palabra 'gustó' pregunta:\n¿quién me importa?",ARCA_BLUE),
           ("2. Key","Cada palabra ofrece una 'etiqueta'.\n'no' tiene una key relevante.",ARCA_ORANGE),
           ("3. Score","query · key (producto punto).\n'gustó'·'no' = alto → mucho peso.",ARCA_RED),
           ("4. Value","Combina los 'value' ponderados.\n'gustó' incorpora a 'no'.",ARCA_GREEN)]
    n=len(steps); w=2.7; gap=(12.5-n*w)/(n+1)
    for i,(t,s,c) in enumerate(steps):
        x=gap+i*(w+gap)
        ax.add_patch(FancyBboxPatch((x,2.1),w,2.2,boxstyle="round,pad=0.05,rounding_size=0.12",facecolor=c,alpha=0.12,edgecolor=c,lw=2))
        ax.text(x+w/2,3.85,t,ha="center",fontsize=11,fontweight="bold",color=c)
        ax.text(x+w/2,2.85,s,ha="center",fontsize=8.7,color="#2D2D2D")
        if i<n-1: ax.add_patch(FancyArrowPatch((x+w,3.2),(x+w+gap,3.2),arrowstyle="-|>",mutation_scale=14,color="#999",lw=1.6))
    ax.text(6.25,1.2,"Query, Key y Value son proyecciones APRENDIDAS del embedding de cada palabra.",ha="center",fontsize=10.5,color=ARCA_DARK,fontweight="bold")
    ax.text(6.25,0.5,"Self-attention = todas las palabras hacen esto a la vez, en paralelo (por eso escala mejor que una LSTM).",ha="center",fontsize=9.5,color="#6B7280")
    save(fig,"fig_qkv.png")

def fig_transformer_arquitectura():
    fig, ax = plt.subplots(figsize=(11,5.6)); ax.set_xlim(0,11); ax.set_ylim(0,7); ax.axis("off"); ax.grid(False)
    ax.text(5.5,6.6,"Un Transformer = embeddings + atención apilada",ha="center",fontsize=12.5,fontweight="bold",color=ARCA_DARK)
    capas=[("Texto → tokens → embeddings","+ posición de cada palabra",ARCA_BLUE,0.4),
           ("Bloque: Self-Attention + Feed-Forward","cada palabra mira a las demás",ARCA_PURPLE,1.55),
           ("× N bloques apilados","representación cada vez más rica",ARCA_PURPLE,2.7),
           ("Cabeza de tarea","clasificar / generar / responder",ARCA_GREEN,3.85)]
    for t,s,c,y in capas:
        ax.add_patch(FancyBboxPatch((1.5,y+0.6),8,0.95,boxstyle="round,pad=0.04,rounding_size=0.1",facecolor=c,alpha=0.12,edgecolor=c,lw=2))
        ax.text(5.5,y+1.22,t,ha="center",fontsize=10.5,fontweight="bold",color=c)
        ax.text(5.5,y+0.82,s,ha="center",fontsize=8.6,color="#6B7280")
    for y in [1.05,2.2,3.35]:
        ax.add_patch(FancyArrowPatch((5.5,y+0.55),(5.5,y+0.72),arrowstyle="-|>",mutation_scale=13,color="#999",lw=1.5))
    ax.text(5.5,0.25,"vs LSTM: procesa todas las palabras EN PARALELO y conecta las lejanas directamente.",ha="center",fontsize=9.5,color=ARCA_RED,fontweight="bold")
    save(fig,"fig_transformer_arquitectura.png")

def fig_atencion_heatmap():
    palabras=["no","me","gustó","nada","la","trama"]; n=len(palabras)
    A=np.array([[0.5,0.1,0.2,0.1,0.05,0.05],[0.2,0.4,0.2,0.1,0.05,0.05],
                [0.55,0.05,0.2,0.15,0.03,0.02],[0.45,0.05,0.25,0.2,0.03,0.02],
                [0.1,0.1,0.1,0.1,0.4,0.2],[0.05,0.05,0.1,0.1,0.2,0.5]])
    fig, ax = plt.subplots(figsize=(6.8,5.6)); im=ax.imshow(A,cmap="Reds",aspect="auto"); ax.grid(False)
    ax.set_xticks(range(n)); ax.set_xticklabels(palabras); ax.set_yticks(range(n)); ax.set_yticklabels(palabras)
    ax.set_xlabel("...mira a esta palabra"); ax.set_ylabel("cada palabra...")
    ax.set_title("Pesos de atención (ilustrativo)\nla fila 'gustó' mira fuerte a 'no'")
    for i in range(n):
        for j in range(n): ax.text(j,i,f"{A[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if A[i,j]>0.4 else "#2D2D2D")
    fig.tight_layout(); save(fig,"fig_atencion_heatmap.png")

def fig_papers_historia():
    fig, ax = plt.subplots(figsize=(12.5,5.5)); ax.set_xlim(0,13); ax.set_ylim(0,7); ax.axis("off"); ax.grid(False)
    ax.text(6.5,6.6,"Papers que cambiaron el mundo",ha="center",fontsize=15,fontweight="bold",color=ARCA_DARK)
    papers=[("1859","Darwin\nOrigen de las especies",ARCA_DARK),("1905","Einstein\nRelatividad especial",ARCA_BLUE),
            ("1948","Shannon\nTeoría de la información",ARCA_PURPLE),("1953","Watson & Crick\nestructura del ADN",ARCA_GREEN),
            ("2017","Vaswani et al.\nAttention Is All You Need",ARCA_RED)]
    ax.plot([0.7,12.3],[3.4,3.4],color="#CCC",lw=2,zorder=0); xs=np.linspace(1.4,11.6,len(papers))
    for i,((yr,txt,c),x) in enumerate(zip(papers,xs)):
        up=i%2==0; y=4.3 if up else 1.7; big=yr=="2017"
        ax.add_patch(Circle((x,3.4),0.13,facecolor=c,edgecolor="white",lw=1.5,zorder=3))
        ax.plot([x,x],[3.4,y+(0.0 if up else 0.95)],color=c,lw=1.2,zorder=1)
        ax.add_patch(FancyBboxPatch((x-1.05,y),2.1,0.95,boxstyle="round,pad=0.04,rounding_size=0.1",facecolor=c,alpha=0.18 if big else 0.10,edgecolor=c,lw=2.5 if big else 1.5,zorder=2))
        ax.text(x,y+0.62,yr,ha="center",fontsize=11,fontweight="bold",color=c)
        ax.text(x,y+0.28,txt,ha="center",va="center",fontsize=8.5,color="#2D2D2D")
    ax.text(6.5,0.6,"El paper de 2017 introdujo el Transformer: la base de ChatGPT, los traductores y casi todo el NLP moderno.",ha="center",fontsize=10,color=ARCA_RED,fontweight="bold")
    save(fig,"fig_papers_historia.png")

def fig_transformer_limite():
    fig, ax = plt.subplots(figsize=(11.5,5)); ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    ax.text(5.5,5.6,"El límite: los transformers de verdad son ENORMES",ha="center",fontsize=13,fontweight="bold",color=ARCA_DARK)
    modelos=[("Nuestro mini\n(notebook)",0.5,ARCA_GREEN,"~0.001"),("BERT base",1.6,ARCA_BLUE,"0.11"),("GPT-3",4.2,ARCA_ORANGE,"175"),("GPT-4 (aprox)",5.0,ARCA_RED,"~1.000+")]
    base=0.8
    for i,(nom,h,c,par) in enumerate(modelos):
        x=1.0+i*2.4; ax.add_patch(Rectangle((x,base),1.5,h,facecolor=c,alpha=0.7))
        ax.text(x+0.75,base+h+0.18,par,ha="center",fontsize=10,fontweight="bold",color=c)
        ax.text(x+0.75,base-0.3,nom,ha="center",va="top",fontsize=9,color="#2D2D2D")
    ax.text(5.5,0.1,"miles de millones de parámetros — NO caben en una GPU normal.",ha="center",fontsize=10,color="#6B7280")
    ax.text(5.5,base+5.4,"Por eso no los entrenamos: usamos modelos PRE-ENTRENADOS, en la nube o vía API (Módulo 6).",ha="center",fontsize=10.5,color=ARCA_PURPLE,fontweight="bold")
    save(fig,"fig_transformer_limite.png")

def fig_decision_tree():
    fig, ax = plt.subplots(figsize=(11.5,5.2)); ax.set_xlim(0,11); ax.set_ylim(0,6); ax.axis("off"); ax.grid(False)
    def node(x,y,w,h,txt,col,fs=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.1",facecolor=col,alpha=0.13,edgecolor=col,lw=2))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color="#2D2D2D")
    node(3.7,4.8,3.6,0.9,"¿Qué necesito del texto?",ARCA_DARK,11)
    node(0.3,2.9,3.0,1.1,"Clasificar rápido\ne interpretar\n→ TF-IDF + LogReg",ARCA_BLUE)
    node(4.0,2.9,3.0,1.1,"Buscar/comparar\npor significado\n→ Embeddings + coseno",ARCA_GREEN)
    node(7.7,2.9,3.0,1.1,"Entender contexto\ncomplejo / generar\n→ Transformers (pre-entrenados)",ARCA_PURPLE)
    for x in [1.8,5.5,9.2]: ax.add_patch(FancyArrowPatch((5.5,4.8),(x,4.0),arrowstyle="-|>",mutation_scale=14,color=ARCA_RED,lw=1.6))
    ax.text(5.5,1.4,"Empieza simple. Sube de complejidad solo cuando lo simple ya no alcanza.",ha="center",fontsize=11,fontweight="bold",color=ARCA_DARK)
    save(fig,"fig_decision_tree.png")


if __name__ == "__main__":
    print("\n--- Vecinos Word2Vec (quality check) ---")
    for w in ["buena","aburrida","excelente","actores","obra"]:
        if w in w2v.wv: print(f"  {w:10}-> "+", ".join(x for x,_ in w2v.wv.most_similar(w,topn=5)))
    print("\nGenerando figuras (solo DATOS reales; los conceptos van en TikZ en las slides)...")
    fig_hero(); fig_problema_descubrimiento(); fig_que_es_nlp()
    fig_eda_ngramas(); fig_sentiment_confusion(); fig_sentiment_coefs()
    fig_zipf(); fig_bow_tfidf(); fig_spam_resultado()
    fig_word2vec_idea(); fig_embeddings_vecinos(); fig_embeddings_2d(); fig_busqueda_coseno()
    fig_atencion_concepto(); fig_qkv(); fig_transformer_arquitectura(); fig_atencion_heatmap()
    fig_papers_historia(); fig_transformer_limite(); fig_decision_tree()
    print(f"\nNumeros reales: sentimiento {acc_sent:.3f} | spam {acc_spam:.3f}")
    print("Listo.")
