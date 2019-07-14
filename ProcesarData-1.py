#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fancyimpute')


# In[ ]:


get_ipython().system('pip install -U -q Pydrive')
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from google.colab import files
from oauth2client.client import GoogleCredentials

#data dependencies
# importando librerias
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import math as mt
import warnings

get_ipython().system('pip install pydotplus')
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import pydotplus
from sklearn import preprocessing


# Ploting styles
# styles: 'fivethirtyeight', 'classic', 'ggplot', 'seaborn-notebook'
# styles: 'seaborn-poster', 'bmh', 'grayscale', 'seaborn-whitegrid'
matplotlib.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print(plt.style.available)
warnings.filterwarnings("ignore")


# In[ ]:


#authenticate
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[ ]:


train_downloaded = drive.CreateFile({'id': '1eLnN7nN8BKCahqqmWKsU1Dp8A4q0D5au'})
train_downloaded.GetContentFile('Bajas20192_null.csv')
df = pd.read_csv('Bajas20192_null.csv')


# In[ ]:


df.ID_BAJA.value_counts()


# In[ ]:


df.Segmento_Planta.value_counts()


# In[ ]:


df.Macrosegmento_Cartera.value_counts()


# In[ ]:


df.Departamento_Alta.value_counts()


# In[ ]:


df.marca.value_counts()


# In[ ]:


marcas = ['SAMSUNG', 'HUAWEI', 'APPLE', 'LENOVO', 'LG', 'MOTOROLA']


# In[ ]:


df.COD_TIPO_CLIENTE.value_counts()


# In[ ]:


def grupo_marcas(marca):
    if marca in marcas:
        return marca
    else:
        return 'OTROS'


# In[ ]:


df['GRUPO_MARCA'] = df['marca'].apply(lambda x: grupo_marcas(x))
df.GRUPO_MARCA.value_counts()


# In[ ]:


df.loc[df.PLAN.isna(), ].head(10)


# In[ ]:


df.shape


# In[ ]:


dfN = df.loc[df.ID_PLAN.notnull(), ]
dfN.shape


# In[ ]:


dfN.ID_BAJA.value_counts()


# In[ ]:


df.ID_BAJA.value_counts()


# In[ ]:


planes = dfN[['ID_PLAN', 'PLAN']]
planes.head()


# In[ ]:


planes.shape


# In[ ]:


planes = planes.drop_duplicates(subset="ID_PLAN")
planes.shape


# In[ ]:


planes.loc[planes.PLAN.isnull(),]


# In[ ]:


dfN = dfN.loc[dfN.ID_PLAN != 0.0,]
dfN.shape


# In[ ]:


dfN.reset_index(drop=True, inplace=True)
dfN.head()


# In[ ]:


dfN["PLAN_GRUPO"] = " "
dfN.loc[dfN['PLAN'].str.contains("PLAN ELIGE", case=False),"PLAN_GRUPO"] = "ELIGE"
dfN.loc[dfN['PLAN'].str.contains("Plan Vuela", case=False),"PLAN_GRUPO"] = "VUELA"
dfN.loc[dfN['PLAN'].str.contains("Plan Staff", case=False),"PLAN_GRUPO"] = "STAFF"
dfN.loc[dfN['PLAN'].str.contains("Plan Portabilidad", case=False),"PLAN_GRUPO"] = "PORTABILIDAD"
dfN.loc[dfN['PLAN'].str.contains("Plan RPM", case=False),"PLAN_GRUPO"] = "RPM"
dfN.loc[dfN['PLAN_GRUPO'] == " ", "PLAN_GRUPO"] = "OTRO"
dfN.PLAN_GRUPO.value_counts()


# In[ ]:


DataUsable = dfN.loc[dfN.ID_BAJA < 2, ]
DataUsable.ID_BAJA.value_counts()


# In[ ]:


columnasModelo1 = ['ENTMIN_MM_MOVISTAR',
       'ENTCANT_LLAM_MM_MOVISTAR', 'ENTCANT_LLAM_DIST_MM_MOVISTAR',
       'ENTMIN_MM_CLARO', 'ENTCANT_LLAM_MM_CLARO',
       'ENTCANT_LLAM_DIST_MM_CLARO', 'ENTMIN_MM_ENTEL',
       'ENTCANT_LLAM_MM_ENTEL', 'ENTCANT_LLAM_DIST_MM_ENTEL',
       'ENTMIN_MM_BITEL', 'ENTCANT_LLAM_MM_BITEL',
       'ENTCANT_LLAM_DIST_MM_BITEL', 'ENTMIN_MM_VIRGIN',
       'ENTCANT_LLAM_MM_VIRGIN', 'ENTCANT_LLAM_DIST_MM_VIRGIN',
       'ENTMIN_MF_MOVISTAR', 'ENTCANT_LLAM_MF_MOVISTAR',
       'ENTCANT_LLAM_DIST_MF_MOVISTAR', 'ENTMIN_MF_OTROS',
       'ENTCANT_LLAM_MF_OTROS', 'ENTCANT_LLAM_DIST_MF_OTROS', 'ENTMIN_LDI',
       'ENTCANT_LLAM_LDI', 'ENTCANT_LLAM_DIST_LDI', 'ENTMIN_OTROS',
       'ENTCANT_LLAM_OTROS', 'ENTCANT_LLAM_DIST_OTROS', 'SALMIN_MM_MOVISTAR',
       'SALCANT_LLAM_MM_MOVISTAR', 'SALCANT_LLAM_DIST_MM_MOVISTAR',
       'SALMIN_MM_CLARO', 'SALCANT_LLAM_MM_CLARO',
       'SALCANT_LLAM_DIST_MM_CLARO', 'SALMIN_MM_ENTEL',
       'SALCANT_LLAM_MM_ENTEL', 'SALCANT_LLAM_DIST_MM_ENTEL',
       'SALMIN_MM_BITEL', 'SALCANT_LLAM_MM_BITEL',
       'SALCANT_LLAM_DIST_MM_BITEL', 'SALMIN_MM_VIRGIN',
       'SALCANT_LLAM_MM_VIRGIN', 'SALCANT_LLAM_DIST_MM_VIRGIN',
       'SALMIN_MF_MOVISTAR', 'SALCANT_LLAM_MF_MOVISTAR',
       'SALCANT_LLAM_DIST_MF_MOVISTAR', 'SALMIN_MF_OTROS',
       'SALCANT_LLAM_MF_OTROS', 'SALCANT_LLAM_DIST_MF_OTROS', 'SALMIN_LDI',
       'SALCANT_LLAM_LDI', 'SALCANT_LLAM_DIST_LDI', 'SALMIN_OTROS',
       'SALCANT_LLAM_OTROS', 'SALCANT_LLAM_DIST_OTROS',
       'GB_FACEBOOK', 'GB_WHATSAPP', 'GB_TWITTER',
       'GB_APPFUTBOL_STREAM', 'GB_APPMUSICA', 'GB_YOUTUBE', 'MB_SKYPE',
       'MB_GOOGLEPLAY', 'GB_NETFLIX', 'GB_SNAPCHAT', 'GB_SPOTIFY',
       'GB_INSTAGRAM', 'GB_WHATSAPPCALL', 'GB_UBER', 'GB_HTTP', 'GB_FREEZONE',
       'GB_TIENDAAPPS', 'GB_GOOGLEMAPS', 'GB_EMAIL', 'GB_LINKEDIN',
       'GB_MOVISTARPLAY','COD_PRODUCTO', 'Tipo_Linea', 'FECHA_ALTA',
       'ID_ESTADO_TELEFONO', 'ID_PLAN', 'PLAN_GRUPO','VALOR_PLAN',
       'SUB_TOTAL_FACT', 'CICLO', 'PRODUCTO'
       'GRUPO_MARCA', 'CF_VOZ', 'CF_DAT', 'Departamento_Alta', 'MACROSEGMENTO',
       'Segmento_Planta', 'Subsegmento_Valor',
       'Subsegmento_Cartera', 'Permanencia_Dias_Inicial',
       'Permanencia_Dias_Transcurridos', 'Permanencia_Dias_Faltantes', 'ID_BAJA',
       'GRUPO_MARCA', 'PLAN_GRUPO']


# In[ ]:


columnasModelo2 = ['ENTMIN_MM_MOVISTAR',
       'ENTCANT_LLAM_MM_MOVISTAR', 'ENTCANT_LLAM_DIST_MM_MOVISTAR',
       'ENTMIN_MM_CLARO', 'ENTCANT_LLAM_MM_CLARO',
       'ENTCANT_LLAM_DIST_MM_CLARO', 'ENTMIN_MM_ENTEL',
       'ENTCANT_LLAM_MM_ENTEL', 'ENTCANT_LLAM_DIST_MM_ENTEL',
       'ENTMIN_MM_BITEL', 'ENTCANT_LLAM_MM_BITEL',
       'ENTCANT_LLAM_DIST_MM_BITEL', 'ENTMIN_MM_VIRGIN',
       'ENTCANT_LLAM_MM_VIRGIN', 'ENTCANT_LLAM_DIST_MM_VIRGIN',
       'ENTMIN_MF_MOVISTAR', 'ENTCANT_LLAM_MF_MOVISTAR',
       'ENTCANT_LLAM_DIST_MF_MOVISTAR', 'ENTMIN_MF_OTROS',
       'ENTCANT_LLAM_MF_OTROS', 'ENTCANT_LLAM_DIST_MF_OTROS', 'ENTMIN_LDI',
       'ENTCANT_LLAM_LDI', 'ENTCANT_LLAM_DIST_LDI', 'ENTMIN_OTROS',
       'ENTCANT_LLAM_OTROS', 'ENTCANT_LLAM_DIST_OTROS', 'SALMIN_MM_MOVISTAR',
       'SALCANT_LLAM_MM_MOVISTAR', 'SALCANT_LLAM_DIST_MM_MOVISTAR',
       'SALMIN_MM_CLARO', 'SALCANT_LLAM_MM_CLARO',
       'SALCANT_LLAM_DIST_MM_CLARO', 'SALMIN_MM_ENTEL',
       'SALCANT_LLAM_MM_ENTEL', 'SALCANT_LLAM_DIST_MM_ENTEL',
       'SALMIN_MM_BITEL', 'SALCANT_LLAM_MM_BITEL',
       'SALCANT_LLAM_DIST_MM_BITEL', 'SALMIN_MM_VIRGIN',
       'SALCANT_LLAM_MM_VIRGIN', 'SALCANT_LLAM_DIST_MM_VIRGIN',
       'SALMIN_MF_MOVISTAR', 'SALCANT_LLAM_MF_MOVISTAR',
       'SALCANT_LLAM_DIST_MF_MOVISTAR', 'SALMIN_MF_OTROS',
       'SALCANT_LLAM_MF_OTROS', 'SALCANT_LLAM_DIST_MF_OTROS', 'SALMIN_LDI',
       'SALCANT_LLAM_LDI', 'SALCANT_LLAM_DIST_LDI', 'SALMIN_OTROS',
       'SALCANT_LLAM_OTROS', 'SALCANT_LLAM_DIST_OTROS',
       'GB_FACEBOOK', 'GB_WHATSAPP', 'GB_TWITTER',
       'GB_APPFUTBOL_STREAM', 'GB_APPMUSICA', 'GB_YOUTUBE', 'MB_SKYPE',
       'MB_GOOGLEPLAY', 'GB_NETFLIX', 'GB_SNAPCHAT', 'GB_SPOTIFY',
       'GB_INSTAGRAM', 'GB_WHATSAPPCALL', 'GB_UBER', 'GB_HTTP', 'GB_FREEZONE',
       'GB_TIENDAAPPS', 'GB_GOOGLEMAPS', 'GB_EMAIL', 'GB_LINKEDIN',
       'GB_MOVISTARPLAY', 'Tipo_Linea', 'VALOR_PLAN', 'CICLO', 'PRODUCTO',
       'Departamento_Alta', 'Subsegmento_Valor',
       'Permanencia_Dias_Inicial', 'Permanencia_Dias_Transcurridos',
       'Permanencia_Dias_Faltantes', 'ID_BAJA', 'GRUPO_MARCA', 'PLAN_GRUPO']


# In[ ]:


DataUsable = DataUsable[columnasModelo2]
DataUsable.shape


# In[ ]:


DataUsable = DataUsable.loc[DataUsable.Permanencia_Dias_Faltantes >0,]
DataUsable.reset_index(drop = True, inplace= True)
DataUsable.head()


# **DATA NUMERICA**

# In[ ]:


numerico = DataUsable.select_dtypes(['int64','float64'])
numerico.head()


# In[ ]:


from fancyimpute import SoftImpute
num_imp = SoftImpute().fit_transform(numerico)


# In[ ]:


num_imp = pd.DataFrame(num_imp)
num_imp.head()


# In[ ]:


num_imp.columns = numerico.columns
num_imp.head()


# In[ ]:


from sklearn import preprocessing
X_num = num_imp.values

normalize = preprocessing.MinMaxScaler()
np_scaled = normalize.fit_transform(X_num)
data_num_imp_n = pd.DataFrame(np_scaled)
data_num_imp_n.head()


# In[ ]:


data_num_imp_n.columns = numerico.columns
data_num_imp_n.head()


# **DATA NO NUMERICA**

# In[ ]:


categorica = DataUsable.select_dtypes([object])
categorica.head()


# In[ ]:


categorica.Departamento_Alta = categorica.Departamento_Alta.fillna(categorica.Departamento_Alta.mode().values[0])
categorica.info()


# In[ ]:


categorica.CICLO.value_counts()


# In[ ]:


categorica["Departamento_Alta"] = categorica["Departamento_Alta"].astype("category").cat.codes
categorica["Subsegmento_Valor"] = categorica["Subsegmento_Valor"].astype("category").cat.codes
categorica["Tipo_Linea"] = categorica["Tipo_Linea"].astype("category").cat.codes


categorica = pd.concat([categorica, pd.get_dummies(categorica['CICLO'], prefix = 'CICLO')], axis=1)
categorica = pd.concat([categorica, pd.get_dummies(categorica['GRUPO_MARCA'], prefix = 'MARCA')], axis=1)
categorica = pd.concat([categorica, pd.get_dummies(categorica['PLAN_GRUPO'], prefix = 'PLAN_GRUPO')], axis=1)

del categorica['CICLO']
del categorica['GRUPO_MARCA']
del categorica['PLAN_GRUPO']


categorica.head()


# In[ ]:


data_num_imp_n.reset_index(drop=True, inplace=True)
categorica.reset_index(drop=True, inplace=True)
datos_procesado = pd.concat([categorica, data_num_imp_n], axis = 1)

datos_procesado.head()


# In[ ]:


datos_procesados = datos_procesado.loc[datos_procesado.PRODUCTO == 'MOVIL', ]
datos_procesados.columns


# In[ ]:


columnasModelo3 = ['Tipo_Linea', 'Departamento_Alta', 'Subsegmento_Valor',
       'CICLO_C05', 'CICLO_C15', 'CICLO_C23', 'MARCA_APPLE', 'MARCA_HUAWEI',
       'MARCA_LENOVO', 'MARCA_LG', 'MARCA_MOTOROLA', 'MARCA_OTROS',
       'MARCA_SAMSUNG', 'PLAN_GRUPO_ELIGE', 'PLAN_GRUPO_OTRO',
       'PLAN_GRUPO_PORTABILIDAD', 'PLAN_GRUPO_RPM', 'PLAN_GRUPO_STAFF',
       'PLAN_GRUPO_VUELA', 'ENTMIN_MM_MOVISTAR', 'ENTCANT_LLAM_MM_MOVISTAR',
       'ENTCANT_LLAM_DIST_MM_MOVISTAR', 'ENTMIN_MM_CLARO',
       'ENTCANT_LLAM_MM_CLARO', 'ENTCANT_LLAM_DIST_MM_CLARO',
       'ENTMIN_MM_ENTEL', 'ENTCANT_LLAM_MM_ENTEL',
       'ENTCANT_LLAM_DIST_MM_ENTEL', 'ENTMIN_MM_BITEL',
       'ENTCANT_LLAM_MM_BITEL', 'ENTCANT_LLAM_DIST_MM_BITEL',
       'ENTMIN_MM_VIRGIN', 'ENTCANT_LLAM_MM_VIRGIN',
       'ENTCANT_LLAM_DIST_MM_VIRGIN', 'ENTMIN_MF_MOVISTAR',
       'ENTCANT_LLAM_MF_MOVISTAR', 'ENTCANT_LLAM_DIST_MF_MOVISTAR',
       'ENTMIN_MF_OTROS', 'ENTCANT_LLAM_MF_OTROS',
       'ENTCANT_LLAM_DIST_MF_OTROS', 'ENTMIN_LDI', 'ENTCANT_LLAM_LDI',
       'ENTCANT_LLAM_DIST_LDI', 'ENTMIN_OTROS', 'ENTCANT_LLAM_OTROS',
       'ENTCANT_LLAM_DIST_OTROS', 'SALMIN_MM_MOVISTAR',
       'SALCANT_LLAM_MM_MOVISTAR', 'SALCANT_LLAM_DIST_MM_MOVISTAR',
       'SALMIN_MM_CLARO', 'SALCANT_LLAM_MM_CLARO',
       'SALCANT_LLAM_DIST_MM_CLARO', 'SALMIN_MM_ENTEL',
       'SALCANT_LLAM_MM_ENTEL', 'SALCANT_LLAM_DIST_MM_ENTEL',
       'SALMIN_MM_BITEL', 'SALCANT_LLAM_MM_BITEL',
       'SALCANT_LLAM_DIST_MM_BITEL', 'SALMIN_MM_VIRGIN',
       'SALCANT_LLAM_MM_VIRGIN', 'SALCANT_LLAM_DIST_MM_VIRGIN',
       'SALMIN_MF_MOVISTAR', 'SALCANT_LLAM_MF_MOVISTAR',
       'SALCANT_LLAM_DIST_MF_MOVISTAR', 'SALMIN_MF_OTROS',
       'SALCANT_LLAM_MF_OTROS', 'SALCANT_LLAM_DIST_MF_OTROS', 'SALMIN_LDI',
       'SALCANT_LLAM_LDI', 'SALCANT_LLAM_DIST_LDI', 'SALMIN_OTROS',
       'SALCANT_LLAM_OTROS', 'SALCANT_LLAM_DIST_OTROS', 'GB_FACEBOOK',
       'GB_WHATSAPP', 'GB_TWITTER', 'GB_APPFUTBOL_STREAM', 'GB_APPMUSICA',
       'GB_YOUTUBE', 'MB_SKYPE', 'MB_GOOGLEPLAY', 'GB_NETFLIX', 'GB_SNAPCHAT',
       'GB_SPOTIFY', 'GB_INSTAGRAM', 'GB_WHATSAPPCALL', 'GB_UBER', 'GB_HTTP',
       'GB_FREEZONE', 'GB_TIENDAAPPS', 'GB_GOOGLEMAPS', 'GB_EMAIL',
       'GB_LINKEDIN', 'GB_MOVISTARPLAY', 'VALOR_PLAN',
       'Permanencia_Dias_Inicial', 'Permanencia_Dias_Transcurridos',
       'Permanencia_Dias_Faltantes']


# In[ ]:


datos_procesado.shape


# In[ ]:


datos_procesado.ID_BAJA.value_counts()


# In[ ]:


datos_procesado[columnasModelo3 + ['ID_BAJA']].to_csv("data_modelo.csv", index=False)
files.download("data_modelo.csv")


# **SELECCION DE VARIABLES**

# In[ ]:


datos_X = datos_procesado[columnasModelo3]
datos_y = datos_procesado["ID_BAJA"]


# In[ ]:


datos_X.shape


# In[ ]:


seed = 77
from xgboost import XGBClassifier
xgbcx = XGBClassifier(random_state = seed)
xgbcx.fit(datos_X, datos_y)


# In[ ]:


IxV = pd.DataFrame({'variable':columnasModelo3,'importancia':xgbcx.feature_importances_})
IxV = IxV.sort_values(by=['importancia'],ascending=False)
IxV.head(10)


# In[ ]:


variables = IxV[:12]
X2 = datos_X[variables.variable]


# **EVALUAR MODELOS**

# In[ ]:


# Usando Train/Test Split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
# separando a partir del Train un train/test split
X_train, X_test, y_train, y_test = train_test_split(X2, datos_y, test_size=0.2, random_state=1)


# In[ ]:


resultados = {}
from sklearn.model_selection import cross_validate

def evaluar_modelo(estimador, X, y):
    resultados_estimador = cross_validate(estimador, X, y,
                     scoring="roc_auc", n_jobs=1, cv=5, return_train_score=True)
    return resultados_estimador

def ver_resultados():
    resultados_df  = pd.DataFrame(resultados).T
    resultados_cols = resultados_df.columns
    for col in resultados_df:
        resultados_df[col] = resultados_df[col].apply(np.mean)
        resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()
    return resultados_df


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


resultados["reg_logistica"] = evaluar_modelo(LogisticRegression(), X2, datos_y)
resultados["naive_bayes"] = evaluar_modelo(GaussianNB(), X2, datos_y)
resultados["rf"] = evaluar_modelo(RandomForestClassifier(), X2, datos_y)
resultados["svc"] = evaluar_modelo(SVC(), X2, datos_y)
resultados["tree"] = evaluar_modelo(DecisionTreeClassifier(), X2, datos_y)


# In[ ]:


ver_resultados()


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# **RANDOM FOREST**

# In[ ]:


estimador_rf = RandomForestClassifier()


# In[ ]:


parametros_busqueda_rf = {
    "criterion": ["gini", "entropy"],
    "n_estimators": np.linspace(1,40,40).astype(int),
    "class_weight": [None, "balanced"]
}


# In[ ]:


grid = GridSearchCV(estimator=estimador_rf, 
                    param_grid=parametros_busqueda_rf,
                    scoring="roc_auc", n_jobs=-1)


# In[ ]:


grid.fit(X2, datos_y)


# In[ ]:


print(grid.best_score_)
print(grid.best_estimator_)


# In[ ]:


modelo_rf = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='entropy', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=40, n_jobs=1, oob_score=False,
                       random_state=None, verbose=0, warm_start=False)
modelo_rf.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(datos_y, modelo_rf.predict(X2))


# In[ ]:


get_ipython().system('pip install joblib')
import joblib

joblib.dump(modelo_rf,"modelo_rf.pkl")
files.download("modelo_rf.pkl")


# In[ ]:


X2.to_csv("data_rf.csv",index=False)
files.download("data_rf.csv")


# **NAIVE BAYES**

# In[ ]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


skf = StratifiedKFold(n_splits=10)
params = {}
nb = GaussianNB()
gs = GridSearchCV(nb, cv=skf, param_grid=params, return_train_score=True, scoring="roc_auc", n_jobs=-1)


# In[ ]:


gs.fit(X_train, y_train)

gs.cv_results_


# In[ ]:


nb.fit(X2, datos_y)
nb.score(X_test, y_test)


# In[ ]:


gs.score(X_test, y_test)


# In[ ]:


gs.cv_results_

