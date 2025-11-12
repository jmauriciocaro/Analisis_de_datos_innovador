# 📊 Predicción Energética Colombia 2022-2030

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Prophet](https://img.shields.io/badge/Prophet-1.1.5-orange.svg)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de análisis predictivo del mercado energético colombiano utilizando técnicas avanzadas de Machine Learning y series temporales para proyectar la generación y demanda de energía hasta 2030.

![Proyección Energética](proyeccion_energia_2030.png)

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Resultados](#resultados)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Metodología](#metodología)
- [Tecnologías](#tecnologías)
- [Autor](#autor)
- [Licencia](#licencia)

---

## 🎯 Descripción

Este proyecto implementa un **pipeline automatizado** para la predicción de generación y demanda de energía eléctrica en Colombia, utilizando datos históricos del Sistema Interconectado Nacional (SIN) desde 2022 hasta 2025, proyectando hasta 2030.

### Objetivos

1. Predecir la **generación diaria de energía** con alta precisión (R² = 0.67)
2. Estimar la **demanda energética** mediante método proxy validado
3. Analizar el **balance energético** y tendencias del mercado
4. Proporcionar herramientas de visualización para toma de decisiones

---

## ✨ Características

- ✅ **Pipeline automatizado completo** desde datos crudos hasta predicciones
- ✅ **Detección y corrección de outliers** con método IQR
- ✅ **3 modelos comparados**: Prophet, Random Forest, Regresión Lineal
- ✅ **Validación cruzada temporal** para garantizar robustez
- ✅ **Proyecciones diarias** hasta 2030-12-31 (1,905 días)
- ✅ **Visualizaciones profesionales** de alta resolución
- ✅ **Exportación automática** de resultados en CSV
- ✅ **Documentación técnica completa**

---

## 📈 Resultados

### Generación de Energía

| Métrica | Valor |
|---------|-------|
| **Modelo** | Prophet |
| **R²** | 0.6726 |
| **MAE** | 5.76 GWh (~2.4% error) |
| **Rango predicciones** | 208 - 255 GWh/día |
| **Promedio 2030** | 245 GWh/día |

### Demanda de Energía

| Métrica | Valor |
|---------|-------|
| **Método** | Proxy (ratio histórico) |
| **Ratio D/G** | 0.9688 (96.88%) |
| **MAE validación** | 6.96 GWh (3.15% error) |
| **R² validación** | 0.2904 ✅ |
| **Pérdidas técnicas** | 3.12% (estándar SIN) |

### Balance Energético

- **Excedente promedio:** 4 GWh/día
- **Cobertura:** 100% (generación > demanda)
- **Tendencia:** Crecimiento sostenido ~1.5% anual

---

## 🚀 Instalación

### Requisitos Previos

- Python >= 3.8
- pip >= 21.0
- ~500 MB espacio en disco

### Paso 1: Clonar el repositorio
(https://github.com/jmauriciocaro/Analisis_de_datos_innovador.git) prediccion-energia-colombia

### Paso 2: Crear entorno virtual
python -m venv .venv
En macOS/Linux:
source .venv/bin/activate
En Windows:
.venv\Scripts\activate


### Paso 3: Instalar dependencias
pip install -r requirements.txt


### Paso 4: Verificar instalación
python -c “import pandas, prophet, sklearn; print(‘✅ Instalación exitosa’)”



---

## 💻 Uso

### Ejecución básica

Importar módulo principal
from pipeline import pipeline_completo
Cargar datos crudos
import pandas as pd df_generacion = pd.read_csv(‘datos/generacion_raw.csv’) df_demanda = pd.read_csv(‘datos/demanda_raw.csv’)
Ejecutar pipeline completo
resultados = pipeline_completo(df_generacion, df_demanda)


### Salida esperada

El pipeline generará automáticamente:

1. **Archivos CSV:**
   - `predicciones_generacion_2030.csv`
   - `predicciones_demanda_2030.csv`
   - `predicciones_energia_2030_completo.csv`

2. **Visualización:**
   - `proyeccion_energia_2030.png` (alta resolución)

3. **Reportes en consola:**
   - Métricas de evaluación
   - Validación cruzada
   - Resumen ejecutivo

---

## 📁 Estructura del Proyecto


Faltaaaa

## 🔬 Metodología

### 1. Preparación de Datos

- **Agregación temporal:** Suma diaria de valores horarios/subhorarios
- **Normalización:** Conversión de kWh a GWh
- **Filtrado temporal:** Datos desde 2022-01-01
- **Detección de outliers:** Método IQR (factor = 3)
- **Corrección:** Reemplazo por mediana

### 2. Modelado de Generación

Se compararon 3 modelos de ML:

| Modelo | R² | MAE | Seleccionado |
|--------|----|----|--------------|
| **Prophet** | **0.67** | **5.76 GWh** | ✅ |
| Random Forest | -0.001 | 11.20 GWh | ❌ |
| Regresión Lineal | -0.009 | 11.27 GWh | ❌ |

**Prophet** captura exitosamente:
- Estacionalidad semanal (variación día laboral vs fin de semana)
- Estacionalidad anual (picos en diciembre)
- Días festivos colombianos
- Tendencia de largo plazo

### 3. Modelado de Demanda

**Método Proxy** basado en principio físico del balance energético:
Demanda = Generación × Ratio_histórico


Donde:
- **Ratio histórico:** 0.9688 (estable ±0.0045)
- **Fundamentación:** Demanda ≈ Generación - Pérdidas técnicas
- **Validación:** R² = 0.29, MAE = 6.96 GWh

Este método supera modelos directos (que presentaban R² negativos) debido a que:
- La demanda tiene alto componente estocástico no predecible
- El ratio D/G es extremadamente estable temporalmente
- Evita sobreajuste a ruido aleatorio

### 4. Validación

**Validación cruzada temporal (75% train / 25% test):**
- Período entrenamiento: 2022-01-01 a 2024-11-01
- Período prueba: 2024-11-02 a 2025-10-13
- Métricas calculadas sobre datos no vistos

---

## 🛠 Tecnologías

### Librerías Principales

- **pandas 2.0+** - Manipulación de datos
- **numpy 1.24+** - Operaciones numéricas
- **prophet 1.1.5** - Modelado de series temporales
- **scikit-learn 1.3+** - Machine Learning
- **matplotlib 3.7+** - Visualización

### Arquitectura

- **Prophet:** Modelo aditivo GAM (Generalized Additive Model)
- **Random Forest:** 100 árboles, profundidad máx. 15
- **Validación:** TimeSeriesSplit con ventanas deslizantes

---

## 👤 Autor

 * Julián Mauricio Caro Correa
 * Lina
 * Liliana
 * Santiago
 * Yan

- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- Email: tu.email@ejemplo.com

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- **XM S.A. E.S.P.** - Datos del Sistema Interconectado Nacional
- **UPME** - Unidad de Planeación Minero Energética
- **Comunidad Prophet** - Documentación y soporte

---

## 📊 Citación

Si utilizas este proyecto en tu investigación o trabajo, por favor cita:

@software{prediccion_energia_colombia_2025, author = {Tu Nombre}, title = {Predicción Energética Colombia 2022-2030}, year = {2025}, url = {https://github.com/tu-usuario/prediccion-energia-colombia} }


---

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:

- Abrir un [Issue](https://github.com/tu-usuario/prediccion-energia-colombia/issues)
- Enviar un [Pull Request](https://github.com/tu-usuario/prediccion-energia-colombia/pulls)
- Contacto directo: tu.email@ejemplo.com

---

<p align="center">
  Hecho con ❤️ y ☕ en Colombia
</p>
