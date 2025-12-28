import streamlit as st
import pandas as pd
import joblib
import time

#Configuracion de la pagina
#st.set_page_config(page_title="SISTEMA MEDICO CON MACHINE LEARNING",page_icon="🏥")
#titulo 
#st.title("Sistema de diagnostico")
#st.markdown("------")

#with st.sidebar:
#    st.header("PANEL E CONTROL")
#    st.info("ESTE SISTEMA ES UN PROTOTIPO DE INFORMACION")

#col1,col2=st.columns(2)
#with col1:
 #   st.subheader("DATOS DEL PACIENTE")
 #   edad=st.number_input("Edad",18,100,30)   
  #  glucosa=st.number_input("NIVEL DE GLUCOSA (mg/dl)",50,500,100) 

#with col2:
#    st.subheader("SIGNOS VITALES")
#    presion=st.slider("Presion arterial",80,200,120) 
#    peso=st.number_input("peso(kg)",40.0,150.0,70.0)

#if st.button("Analizar riesgo", type="primary") :
#    with st.spinner("Analizando patrones con inteligencia artificial..."):
#        time.sleep(2)

#    if glucosa>140:
#        st.error("Alerta : Posible riesgo detectado")
#        st.write(f"El paciente de {edad} años muestra niveles atipicos.")

#    else:
#        st.success("RESULTADO:PACIENTE SALUDABLE")
#        st.write("Todos los parametros estan en orden.")  
# 
# 

#CONFIGURACIÓN

st.set_page_config(page_title="SISTEMA MEDICO IA", page_icon="🏥",layout="centered")

#---CARGAMOS EL MODELO
try:
    modelo=joblib.load('modelo_diabetes.pkl')
    st.toast("Modelo IA cargado exitosamente", icon='🧠')
except:
    st.error("No se encontro el archivo 'modelo_diabetes.pkl'") 
    st.stop()

#INTERFAZ
# 
with st.sidebar:
    st.header("PANEL MÉDICO")
    st.write("Este sistema utiliza Regresion Logistica entrenada con el dataset")

st.title("DIAGNÓSTICO PREDICTIVO REAL")
st.markdown("-------------------")

col1,col2=st.columns(2)

with col1:
    st.subheader("DATOS CLÍNICOS")
    glucosa=st.number_input("Nivel de Glucosa (mg/dL)",50,200,100)
    edad=st.number_input("Edad",21,100,30)

with col2:
    st.subheader("Fisiologia")
    presion=st.slider("Presión arterial",40,140,70)
    #calculamos el IMC
    peso=st.number_input("Peso(kg)",40.0,150.0,70.0)
    altura=st.number_input("Altura (m)",1.40,2.20,1.70) 
    imc=peso/(altura**2)
    st.info(f"IMC Calculado: {imc:.2f}")  

#Boton de prediccion
if st.button("Ejecutar diagnostico IA",type="primary"):
    with st.spinner("El modelo esta evaluando los patrones...") :
        time.sleep(1)

    datos_paciente=pd.DataFrame([[glucosa,presion,imc,edad]],
                                columns=['Glucosa','Presion','IMC','Edad']) 

    prediccion=modelo.predict(datos_paciente)[0]
    probabilidad=modelo.predict_proba(datos_paciente)[0][1] 

    if prediccion==1:
        st.error(f"Riesgo detectado (Probabilidad: {probabilidad*100:.1f}%)") 
        st.write("El modelo sugiere alta probabilidad de diabetes basada en los patrones aprendidos.") 

    else:
        st.success(f"Paciente SANO (Probabilidad de riesgo:{probabilidad*100:.1f}%)") 
        st.write("Los parametros ingresados no coinceden con los patrones diabéticos del dataset.")       