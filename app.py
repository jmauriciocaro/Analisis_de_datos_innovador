import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción Energética Colombia 2030",
    page_icon="⚡",
    layout="wide"
)

# Título principal
st.title("⚡ Predicción Energética Colombia 2025-2030")
st.markdown("---")

# =============================================================================
# FUNCIONES
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga los datasets de generación, capacidad y demanda"""
    df_generacion = pd.read_csv('data/Generacion.csv')
    df_capacidad = pd.read_csv('data/Capacidad.csv')
    df_demanda = pd.read_csv('data/Demanda.csv')
    return df_generacion, df_capacidad, df_demanda

def preparar_datos_diarios(df, fecha_col='Fecha', valor_col='GeneracionRealEstimada'):
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df['Fecha_Dia'] = df[fecha_col].dt.date
    resultado = df.groupby('Fecha_Dia')[valor_col].sum().reset_index()
    resultado.columns = ['Fecha', 'Valor']
    resultado['Fecha'] = pd.to_datetime(resultado['Fecha'])
    resultado = resultado.sort_values('Fecha').reset_index(drop=True)
    return resultado

def normalizar_a_gwh(df, valor_col='Valor'):
    df = df.copy()
    df['Valor_GWh'] = df[valor_col] / 1_000_000
    df = df[['Fecha', 'Valor_GWh']]
    return df

def filtrar_por_fecha(df, fecha_inicio='2022-01-01', fecha_fin=None):
    df = df.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    if fecha_inicio:
        df = df[df['Fecha'] >= fecha_inicio]
    if fecha_fin:
        df = df[df['Fecha'] <= fecha_fin]
    return df.reset_index(drop=True)

def corregir_outliers(df, col_valor='Valor_GWh', metodo='mediana', factor=3):
    df = df.copy()
    Q1 = df[col_valor].quantile(0.25)
    Q3 = df[col_valor].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    if metodo == 'mediana':
        valor_reemplazo = df[col_valor].median()
    else:
        valor_reemplazo = df[col_valor].mean()
    
    mask_outliers = (df[col_valor] < lower) | (df[col_valor] > upper)
    df.loc[mask_outliers, col_valor] = valor_reemplazo
    return df

def preparar_datos_ml_diario(df, col_fecha='Fecha', col_valor='Valor_GWh'):
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.sort_values(col_fecha).reset_index(drop=True)
    fecha_inicio = df[col_fecha].min()
    df['dias_desde_inicio'] = (df[col_fecha] - fecha_inicio).dt.days
    X = df[['dias_desde_inicio']].values
    y = df[col_valor].values
    return X, y, df, fecha_inicio

def preparar_datos_prophet(df, col_fecha='Fecha', col_valor='Valor_GWh'):
    df = df.copy()
    df['ds'] = pd.to_datetime(df[col_fecha])
    df['y'] = df[col_valor]
    return df[['ds', 'y']]

def entrenar_prophet_generacion(df):
    df_prophet = preparar_datos_prophet(df)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    model.add_country_holidays(country_name='CO')
    model.fit(df_prophet)
    return model

def predecir_hasta_2030(model, df_prep, fecha_inicio, tipo='prophet'):
    ultima_fecha = pd.to_datetime(df_prep['Fecha'].max())
    fecha_final = pd.Timestamp('2030-12-31')
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), end=fecha_final, freq='D')
    
    if tipo == 'prophet':
        future = pd.DataFrame({'ds': fechas_futuras})
        forecast = model.predict(future)
        predicciones = forecast['yhat'].values
    else:
        ultimo_dia = df_prep['dias_desde_inicio'].max()
        dias_futuros = np.arange(ultimo_dia + 1, ultimo_dia + 1 + len(fechas_futuras)).reshape(-1, 1)
        predicciones = model.predict(dias_futuros)
    
    df_futuro = pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion_GWh': predicciones})
    return df_futuro

@st.cache_data
def comparar_modelos(df, col_fecha='Fecha', col_valor='Valor_GWh'):
    X, y, df_prep, fecha_inicio = preparar_datos_ml_diario(df, col_fecha, col_valor)
    df_prophet = preparar_datos_prophet(df, col_fecha, col_valor)
    
    split = len(X) - 90
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    resultados = {}
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    resultados['Random Forest'] = {
        'modelo': rf,
        'metricas': {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred)
        },
        'tipo': 'ml'
    }
    
    # Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    resultados['Regresion Lineal'] = {
        'modelo': lr,
        'metricas': {
            'MAE': mean_absolute_error(y_test, lr_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'R2': r2_score(y_test, lr_pred)
        },
        'tipo': 'ml'
    }
    
    # Prophet
    prophet = entrenar_prophet_generacion(df.iloc[:split])
    future_test = pd.DataFrame({'ds': df_prophet.iloc[split:]['ds']})
    prophet_forecast = prophet.predict(future_test)
    prophet_pred = prophet_forecast['yhat'].values
    resultados['Prophet'] = {
        'modelo': prophet,
        'metricas': {
            'MAE': mean_absolute_error(y_test, prophet_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, prophet_pred)),
            'R2': r2_score(y_test, prophet_pred)
        },
        'tipo': 'prophet'
    }
    
    mejor_nombre = min(resultados, key=lambda x: resultados[x]['metricas']['MAE'])
    mejor_info = resultados[mejor_nombre]
    
    if mejor_info['tipo'] == 'prophet':
        mejor_modelo_final = entrenar_prophet_generacion(df)
    elif mejor_nombre == 'Random Forest':
        mejor_modelo_final = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
        mejor_modelo_final.fit(X, y)
    else:
        mejor_modelo_final = LinearRegression()
        mejor_modelo_final.fit(X, y)
    
    predicciones = predecir_hasta_2030(mejor_modelo_final, df_prep, fecha_inicio, mejor_info['tipo'])
    
    return mejor_nombre, mejor_modelo_final, predicciones, resultados

def estimar_demanda_proxy(pred_gen, df_demanda_hist, df_gen_hist):
    pred_demanda = pred_gen.copy()
    df_hist_merged = pd.merge(
        df_gen_hist[['Fecha', 'Valor_GWh']],
        df_demanda_hist[['Fecha', 'Valor_GWh']],
        on='Fecha',
        suffixes=('_gen', '_dem')
    )
    ratio = df_hist_merged['Valor_GWh_dem'].mean() / df_hist_merged['Valor_GWh_gen'].mean()
    pred_demanda['Prediccion_GWh'] = pred_demanda['Prediccion_GWh'] * ratio
    return pred_demanda, ratio

def validacion_simple_proxy(df_gen, df_dem):
    """
    Validación 75% train / 25% test para método proxy.
    Alinea fechas entre generación y demanda.
    """
    # Alinear fechas
    df_gen = df_gen.sort_values('Fecha').reset_index(drop=True)
    df_dem = df_dem.sort_values('Fecha').reset_index(drop=True)
    
    # Merge por fecha para asegurar alineación
    df_merged = pd.merge(df_gen, df_dem, on='Fecha', suffixes=('_gen', '_dem'))
    
    # Split 75/25
    split = int(len(df_merged) * 0.75)
    
    train = df_merged.iloc[:split]
    test = df_merged.iloc[split:]
    
    # Calcular ratio
    ratio = train['Valor_GWh_dem'].mean() / train['Valor_GWh_gen'].mean()
    
    # Predecir
    dem_pred = test['Valor_GWh_gen'].values * ratio
    
    # Evaluar
    mae = mean_absolute_error(test['Valor_GWh_dem'], dem_pred)
    rmse = np.sqrt(mean_squared_error(test['Valor_GWh_dem'], dem_pred))
    r2 = r2_score(test['Valor_GWh_dem'], dem_pred)
    
    return {
        'ratio': ratio,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'train_dates': (train['Fecha'].min(), train['Fecha'].max()),
        'test_dates': (test['Fecha'].min(), test['Fecha'].max()),
        'dias_alineados': len(df_merged),
        'error_relativo': 100*mae/test['Valor_GWh_dem'].mean()
    }

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("👥 Equipo de Desarrollo")
    
    st.markdown("### Integrantes:")
    st.markdown("* **Julián Caro** - [@jmauriciocaro](https://github.com/jmauriciocaro)")
    st.markdown("* **Liliana Correa** - [@liliana1411](https://github.com/liliana1411)")
    st.markdown("* **Lina Ramírez** - [@linaramirezbootcamp-dot](https://github.com/linaramirezbootcamp-dot)")
    st.markdown("* **Yan Hoyos** - [@seyanhc](https://github.com/seyanhc)")
    st.markdown("* **Santiago Arboleda** - [@santiagoarbolpiedra](https://github.com/santiagoarbolpiedra)")
    
    st.markdown("---")
    st.markdown("### 📂 Repositorio del Proyecto")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Proyecto-blue?logo=github)](https://github.com/jmauriciocaro/Analisis_de_datos_innovador)")
    
    st.markdown("---")
    st.markdown("### 📊 Secciones")
    seccion = st.radio(
        "Ir a:",
        ["Datos Crudos", "Modelado", "Predicciones", "Análisis"]
    )

# Configuración fija
fecha_inicio = pd.to_datetime('2022-01-01')
mostrar_outliers = True

# =============================================================================
# CARGA DE DATOS
# =============================================================================

with st.spinner("Cargando datos..."):
    df_generacion, df_capacidad, df_demanda = cargar_datos()

# =============================================================================
# SECCIÓN: DATOS CRUDOS
# =============================================================================

if seccion == "Datos Crudos":
    st.header("📊 Datos Crudos")
    
    tab1, tab2, tab3 = st.tabs(["Generación", "Capacidad", "Demanda"])
    
    with tab1:
        st.subheader("Generación")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total registros", f"{len(df_generacion):,}")
        with col2:
            st.metric("Columnas", len(df_generacion.columns))
        with col3:
            st.metric("Valores nulos", df_generacion.isnull().sum().sum())
        st.dataframe(df_generacion, use_container_width=True)
    
    with tab2:
        st.subheader("Capacidad")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total registros", f"{len(df_capacidad):,}")
        with col2:
            st.metric("Columnas", len(df_capacidad.columns))
        with col3:
            st.metric("Valores nulos", df_capacidad.isnull().sum().sum())
        st.dataframe(df_capacidad, use_container_width=True)
    
    with tab3:
        st.subheader("Demanda")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total registros", f"{len(df_demanda):,}")
        with col2:
            st.metric("Columnas", len(df_demanda.columns))
        with col3:
            st.metric("Valores nulos", df_demanda.isnull().sum().sum())
        st.dataframe(df_demanda, use_container_width=True)

# =============================================================================
# SECCIÓN: MODELADO
# =============================================================================

elif seccion == "Modelado":
    st.header("🤖 Entrenamiento de Modelos")
    
    if st.button("▶️ Ejecutar Pipeline de Modelado", type="primary"):
        with st.spinner("Procesando datos..."):
            # Preparar generación
            df_gen_diario = preparar_datos_diarios(df_generacion, 'Fecha', 'GeneracionRealEstimada')
            df_gen_gwh = normalizar_a_gwh(df_gen_diario)
            df_gen_gwh = filtrar_por_fecha(df_gen_gwh, str(fecha_inicio))
            
            # Preparar demanda
            df_dem_diario = preparar_datos_diarios(df_demanda, 'FechaHora', 'Valor')
            df_dem_gwh = normalizar_a_gwh(df_dem_diario)
            df_dem_gwh = filtrar_por_fecha(df_dem_gwh, str(fecha_inicio))
            
            if mostrar_outliers:
                df_dem_gwh = corregir_outliers(df_dem_gwh)
            
            st.success("✓ Datos preparados")
        
        with st.spinner("Entrenando modelos..."):
            mejor_modelo_gen, modelo_gen, pred_gen, resultados_gen = comparar_modelos(df_gen_gwh)
            st.success(f"✓ Mejor modelo: {mejor_modelo_gen}")
        
        # Mostrar resultados
        st.subheader("📈 Comparación de Modelos")
        
        # Crear DataFrame para comparación
        comparacion_data = []
        for nombre, info in resultados_gen.items():
            comparacion_data.append({
                'Modelo': nombre,
                'MAE': info['metricas']['MAE'],
                'RMSE': info['metricas']['RMSE'],
                'R²': info['metricas']['R2']
            })
        df_comparacion = pd.DataFrame(comparacion_data)
        
        # Destacar mejor modelo
        def highlight_best(row):
            if row['Modelo'] == mejor_modelo_gen:
                return ['background-color: #90EE90; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_comparacion.style.apply(highlight_best, axis=1).format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'R²': '{:.4f}'
            }),
            use_container_width=True
        )
        
        st.info(f"🏆 **Mejor Modelo:** {mejor_modelo_gen} (menor MAE)")
        
        # Métricas en columnas
        st.subheader("📊 Métricas Detalladas")
        cols = st.columns(3)
        for i, (nombre, info) in enumerate(resultados_gen.items()):
            with cols[i]:
                # Emoji según rendimiento
                if nombre == mejor_modelo_gen:
                    emoji = "🥇"
                elif info['metricas']['R2'] > 0.5:
                    emoji = "🥈"
                else:
                    emoji = "🥉"
                
                st.metric(f"{emoji} {nombre}", f"R² = {info['metricas']['R2']:.4f}")
                st.write(f"**MAE:** {info['metricas']['MAE']:.4f} GWh")
                st.write(f"**RMSE:** {info['metricas']['RMSE']:.4f} GWh")
                
                # Interpretación
                if info['metricas']['R2'] > 0.5:
                    st.success("✓ Buen ajuste")
                elif info['metricas']['R2'] > 0:
                    st.warning("⚠️ Ajuste moderado")
                else:
                    st.error("❌ Ajuste pobre")
        
        # Gráfica comparativa
        st.subheader("📊 Comparación Visual")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE y RMSE
        modelos = df_comparacion['Modelo']
        x = np.arange(len(modelos))
        width = 0.35
        
        ax1.bar(x - width/2, df_comparacion['MAE'], width, label='MAE', color='#F77F00', alpha=0.8)
        ax1.bar(x + width/2, df_comparacion['RMSE'], width, label='RMSE', color='#D62828', alpha=0.8)
        ax1.set_xlabel('Modelo', fontweight='bold')
        ax1.set_ylabel('Error (GWh)', fontweight='bold')
        ax1.set_title('MAE y RMSE por Modelo', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modelos, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # R²
        colors = ['#90EE90' if m == mejor_modelo_gen else '#2E86AB' for m in modelos]
        ax2.bar(modelos, df_comparacion['R²'], color=colors, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Modelo', fontweight='bold')
        ax2.set_ylabel('R² Score', fontweight='bold')
        ax2.set_title('Coeficiente de Determinación (R²)', fontweight='bold')
        ax2.set_xticklabels(modelos, rotation=15, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Información adicional
        with st.expander("ℹ️ ¿Qué significan estas métricas?"):
            st.markdown("""
            **MAE (Mean Absolute Error):**
            - Error promedio absoluto en GWh
            - Más bajo es mejor
            - Interpretación directa: el modelo se equivoca en promedio X GWh
            
            **RMSE (Root Mean Squared Error):**
            - Penaliza más los errores grandes
            - Más bajo es mejor
            - Sensible a outliers
            
            **R² (Coeficiente de Determinación):**
            - Mide qué tan bien el modelo explica la variabilidad
            - Rango: -∞ a 1
            - R² = 1: predicción perfecta
            - R² = 0: el modelo no es mejor que usar la media
            - R² < 0: el modelo es peor que usar la media
            """)
        
        st.markdown("---")
        
        # Validación del método proxy
        st.subheader("🔍 Validación del Método Proxy para Demanda")
        
        st.info("💡 **Método Proxy:** Se utiliza la relación histórica entre Generación y Demanda para estimar la demanda futura.")
        
        with st.spinner("Validando método proxy..."):
            validacion = validacion_simple_proxy(df_gen_gwh, df_dem_gwh)
        
        # Mostrar resultados de validación
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ratio D/G", f"{validacion['ratio']:.4f}")
        with col2:
            st.metric("MAE", f"{validacion['MAE']:.2f} GWh")
        with col3:
            st.metric("RMSE", f"{validacion['RMSE']:.2f} GWh")
        with col4:
            st.metric("R²", f"{validacion['R2']:.4f}")
        
        # Interpretación del resultado
        if validacion['R2'] > 0.7:
            st.success("✅ Excelente ajuste del método proxy")
        elif validacion['R2'] > 0.5:
            st.success("✅ Buen ajuste del método proxy")
        elif validacion['R2'] > 0.3:
            st.warning("⚠️ Ajuste moderado del método proxy")
        else:
            st.write("⚠️ El ajuste puede mejorarse")
        
        with st.expander("📊 Ver detalles de la validación"):
            st.markdown(f"""
            **Método de Validación: Train/Test Split 75%/25%**
            
            El método proxy estima la demanda utilizando la generación predicha multiplicada por un ratio histórico.
            
            **Datos utilizados:**
            - **Días alineados entre generación y demanda:** {validacion['dias_alineados']} días
            - **Período de entrenamiento:** {validacion['train_dates'][0].strftime('%Y-%m-%d')} a {validacion['train_dates'][1].strftime('%Y-%m-%d')}
            - **Período de prueba:** {validacion['test_dates'][0].strftime('%Y-%m-%d')} a {validacion['test_dates'][1].strftime('%Y-%m-%d')}
            
            **Resultados de la validación:**
            - **Ratio Demanda/Generación:** {validacion['ratio']:.4f}
            - **Error Absoluto Medio (MAE):** {validacion['MAE']:.2f} GWh
            - **Error Relativo:** {validacion['error_relativo']:.2f}%
            - **RMSE:** {validacion['RMSE']:.2f} GWh
            - **R² Score:** {validacion['R2']:.4f}
            
            **Interpretación:**
            - El método proxy usa el ratio histórico: Demanda = Generación × {validacion['ratio']:.4f}
            - Un R² de {validacion['R2']:.4f} indica {"un excelente ajuste" if validacion['R2'] > 0.7 else "un buen ajuste" if validacion['R2'] > 0.5 else "un ajuste moderado" if validacion['R2'] > 0.3 else "que el ajuste puede mejorarse"}
            - El error promedio es de {validacion['MAE']:.2f} GWh ({validacion['error_relativo']:.2f}% del valor real)
            """)
        
        # Guardar en session_state
        st.session_state['df_gen_gwh'] = df_gen_gwh
        st.session_state['df_dem_gwh'] = df_dem_gwh
        st.session_state['pred_gen'] = pred_gen
        st.session_state['modelo_gen'] = modelo_gen
        st.session_state['mejor_modelo'] = mejor_modelo_gen
        st.session_state['resultados_gen'] = resultados_gen
        st.session_state['validacion'] = validacion

# =============================================================================
# SECCIÓN: PREDICCIONES
# =============================================================================

elif seccion == "Predicciones":
    st.header("🔮 Predicciones 2025-2030")
    
    if 'pred_gen' in st.session_state:
        pred_gen = st.session_state['pred_gen']
        df_gen_gwh = st.session_state['df_gen_gwh']
        df_dem_gwh = st.session_state['df_dem_gwh']
        
        # Validación del método proxy
        st.subheader("🔍 Validación del Método Proxy")
        with st.spinner("Validando método proxy..."):
            validacion = validacion_simple_proxy(df_gen_gwh, df_dem_gwh)
        
        # Mostrar resultados de validación
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ratio D/G", f"{validacion['ratio']:.4f}")
        with col2:
            st.metric("MAE", f"{validacion['MAE']:.2f} GWh")
        with col3:
            st.metric("RMSE", f"{validacion['RMSE']:.2f} GWh")
        with col4:
            st.metric("R²", f"{validacion['R2']:.4f}")
        
        with st.expander("📊 Ver detalles de la validación"):
            st.markdown(f"""
            **Método de Validación: Train/Test Split 75%/25%**
            
            - **Días alineados entre generación y demanda:** {validacion['dias_alineados']} días
            - **Período de entrenamiento:** {validacion['train_dates'][0].strftime('%Y-%m-%d')} a {validacion['train_dates'][1].strftime('%Y-%m-%d')}
            - **Período de prueba:** {validacion['test_dates'][0].strftime('%Y-%m-%d')} a {validacion['test_dates'][1].strftime('%Y-%m-%d')}
            
            **Resultados:**
            - **Ratio Demanda/Generación:** {validacion['ratio']:.4f}
            - **Error Absoluto Medio (MAE):** {validacion['MAE']:.2f} GWh
            - **Error Relativo:** {validacion['error_relativo']:.2f}%
            - **RMSE:** {validacion['RMSE']:.2f} GWh
            - **R² Score:** {validacion['R2']:.4f}
            
            **Interpretación:**
            El método proxy utiliza la relación histórica entre generación y demanda para estimar la demanda futura.
            Un R² de {validacion['R2']:.4f} indica {"un buen ajuste" if validacion['R2'] > 0.7 else "un ajuste moderado" if validacion['R2'] > 0.5 else "un ajuste que puede mejorarse"}.
            """)
        
        st.markdown("---")
        
        with st.spinner("Calculando demanda..."):
            pred_dem, ratio = estimar_demanda_proxy(pred_gen, df_dem_gwh, df_gen_gwh)
        
        # Consolidar
        pred_consolidado = pred_gen[['Fecha', 'Prediccion_GWh']].copy()
        pred_consolidado.columns = ['Fecha', 'Generacion_GWh']
        pred_consolidado['Demanda_GWh'] = pred_dem['Prediccion_GWh'].values
        pred_consolidado['Balance_GWh'] = pred_consolidado['Generacion_GWh'] - pred_consolidado['Demanda_GWh']
        
        # Métricas
        st.subheader("📊 Métricas de Predicción")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generación Promedio", f"{pred_consolidado['Generacion_GWh'].mean():.2f} GWh/día")
        with col2:
            st.metric("Demanda Promedio", f"{pred_consolidado['Demanda_GWh'].mean():.2f} GWh/día")
        with col3:
            st.metric("Balance Promedio", f"{pred_consolidado['Balance_GWh'].mean():.2f} GWh/día")
        
        # Gráfica principal
        st.subheader("📊 Proyección Generación vs Demanda")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pred_consolidado['Fecha'], pred_consolidado['Generacion_GWh'], label='Generación', linewidth=1.5, color='#2E86AB')
        ax.plot(pred_consolidado['Fecha'], pred_consolidado['Demanda_GWh'], label='Demanda', linewidth=1.5, color='#F77F00')
        ax.fill_between(pred_consolidado['Fecha'], pred_consolidado['Demanda_GWh'], 
                        pred_consolidado['Generacion_GWh'], alpha=0.3, label='Excedente', color='#06A77D')
        ax.legend(fontsize=11)
        ax.set_ylabel('Energía (GWh/día)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
        ax.set_title('Proyección de Generación y Demanda Energética Colombia 2025-2030', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabla de predicciones
        st.subheader("📋 Tabla de Predicciones")
        st.dataframe(pred_consolidado, use_container_width=True)
        
        # Botón de descarga
        csv = pred_consolidado.to_csv(index=False)
        st.download_button(
            label="💾 Descargar Predicciones CSV",
            data=csv,
            file_name="predicciones_energia_2030.csv",
            mime="text/csv"
        )
        
        st.session_state['pred_consolidado'] = pred_consolidado
        st.session_state['validacion'] = validacion
    
    else:
        st.warning("⚠️ Primero ejecuta el modelado en la sección 'Modelado'")

# =============================================================================
# SECCIÓN: ANÁLISIS
# =============================================================================

elif seccion == "Análisis":
    st.header("📈 Análisis Detallado")
    
    if 'pred_consolidado' in st.session_state:
        df = st.session_state['pred_consolidado'].copy()
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['DiaSemana'] = df['Fecha'].dt.day_name()
        
        # Métricas generales
        st.subheader("📊 Resumen Ejecutivo")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gen. Promedio", f"{df['Generacion_GWh'].mean():.2f} GWh/día")
        with col2:
            st.metric("Dem. Promedio", f"{df['Demanda_GWh'].mean():.2f} GWh/día")
        with col3:
            st.metric("Balance Promedio", f"{df['Balance_GWh'].mean():.2f} GWh/día")
        with col4:
            crecimiento_gen = ((df[df['Año']==2030]['Generacion_GWh'].mean() / 
                               df[df['Año']==2025]['Generacion_GWh'].mean()) - 1) * 100
            st.metric("Crecimiento 2025-2030", f"{crecimiento_gen:.2f}%")
        
        st.markdown("---")
        
        # Análisis anual
        st.subheader("📊 Análisis Anual")
        df_anual = df.groupby('Año').agg({
            'Generacion_GWh': 'mean',
            'Demanda_GWh': 'mean',
            'Balance_GWh': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(df_anual))
        width = 0.25
        bars1 = ax.bar(x - width, df_anual['Generacion_GWh'], width, label='Generación', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x, df_anual['Demanda_GWh'], width, label='Demanda', color='#F77F00', alpha=0.8)
        bars3 = ax.bar(x + width, df_anual['Balance_GWh'], width, label='Balance', color='#06A77D', alpha=0.8)
        
        # Valores en barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Año', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energía Promedio Diaria (GWh)', fontsize=12, fontweight='bold')
        ax.set_title('Tendencia Anual de Generación, Demanda y Balance 2025-2030', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual['Año'])
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Análisis mensual
        st.subheader("📅 Comparación Mensual")
        df_mensual = df.groupby(['Año', 'Mes']).agg({
            'Generacion_GWh': 'sum',
            'Demanda_GWh': 'sum',
            'Balance_GWh': 'sum'
        }).reset_index()
        df_mensual['Periodo'] = pd.to_datetime(
            df_mensual[['Año', 'Mes']].rename(columns={'Año': 'year', 'Mes': 'month'}).assign(day=1)
        )
        
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(df_mensual))
        width = 0.35
        ax.bar(x - width/2, df_mensual['Generacion_GWh'], width, label='Generación', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, df_mensual['Demanda_GWh'], width, label='Demanda', color='#F77F00', alpha=0.8)
        ax.set_xlabel('Periodo', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energía Total (GWh/mes)', fontsize=12, fontweight='bold')
        ax.set_title('Comparación Mensual de Generación vs Demanda 2025-2030', fontsize=13, fontweight='bold')
        ax.set_xticks(x[::3])
        ax.set_xticklabels(df_mensual['Periodo'].dt.strftime('%b %Y')[::3], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Patrón semanal
        st.subheader("📆 Patrón Semanal")
        df_semanal = df.groupby('DiaSemana').agg({
            'Generacion_GWh': 'mean',
            'Demanda_GWh': 'mean'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(dias_es))
        width = 0.35
        bars1 = ax.bar(x - width/2, df_semanal['Generacion_GWh'], width, label='Generación', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, df_semanal['Demanda_GWh'], width, label='Demanda', color='#F77F00', alpha=0.8)
        
        # Valores en barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Día de la Semana', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energía Promedio (GWh/día)', fontsize=12, fontweight='bold')
        ax.set_title('Patrón Semanal de Generación y Demanda', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dias_es, rotation=30, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.subheader("📊 Estadísticas Descriptivas")
        st.dataframe(df[['Generacion_GWh', 'Demanda_GWh', 'Balance_GWh']].describe(), use_container_width=True)
        
        # Información adicional
        with st.expander("📈 Ver Crecimiento Detallado"):
            crecimiento_dem = ((df[df['Año']==2030]['Demanda_GWh'].mean() / 
                               df[df['Año']==2025]['Demanda_GWh'].mean()) - 1) * 100
            
            st.write(f"**Crecimiento de Generación (2025-2030):** +{crecimiento_gen:.2f}%")
            st.write(f"**Crecimiento de Demanda (2025-2030):** +{crecimiento_dem:.2f}%")
            st.write(f"**Excedente Mínimo:** {df['Balance_GWh'].min():.2f} GWh/día")
            st.write(f"**Excedente Máximo:** {df['Balance_GWh'].max():.2f} GWh/día")
    
    else:
        st.warning("⚠️ Primero genera las predicciones en la sección 'Predicciones'")