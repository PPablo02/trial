
import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from datetime import date
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from plotly.subplots import make_subplots


# --- Configuración de los ETFs ---
tickers = {
    "TLT": {
        "nombre": "iShares 20+ Year Treasury Bond ETF",
        "descripcion": "Este ETF sigue el índice ICE U.S. Treasury 20+ Year Bond Index, compuesto por bonos del gobierno de EE. UU. con vencimientos superiores a 20 años.",
        "sector": "Renta fija",
        "categoria": "Bonos del Tesoro de EE. UU.",
        "exposicion": "Bonos del gobierno de EE. UU. a largo plazo.",
        "moneda": "USD",
        "beta": 0.2,
        "top_holdings": [
            {"symbol": "US Treasury", "holdingPercent": "100%"}
        ],
        "gastos": "0.15%",
        "rango_1y": "120-155 USD",
        "rendimiento_ytd": "5%",
        "duracion": "Larga"
    },
    "EMB": {
        "nombre": "iShares JP Morgan USD Emerging Markets Bond ETF",
        "descripcion": "Este ETF sigue el índice J.P. Morgan EMBI Global Diversified Index, que rastrea bonos soberanos de mercados emergentes en dólares estadounidenses.",
        "sector": "Renta fija",
        "categoria": "Bonos emergentes",
        "exposicion": "Bonos soberanos de mercados emergentes denominados en USD.",
        "moneda": "USD",
        "beta": 0.6,
        "top_holdings": [
            {"symbol": "Brazil 10Yr Bond", "holdingPercent": "10%"},
            {"symbol": "Mexico 10Yr Bond", "holdingPercent": "9%"},
            {"symbol": "Russia 10Yr Bond", "holdingPercent": "7%"}
        ],
        "gastos": "0.39%",
        "rango_1y": "85-105 USD",
        "rendimiento_ytd": "8%",
        "duracion": "Media"
    },
    "SPY": {
        "nombre": "SPDR S&P 500 ETF Trust",
        "descripcion": "Este ETF sigue el índice S&P 500, compuesto por las 500 principales empresas de EE. UU.",
        "sector": "Renta variable",
        "categoria": "Acciones grandes de EE. UU.",
        "exposicion": "Acciones de las 500 empresas más grandes de EE. UU.",
        "moneda": "USD",
        "beta": 1.0,
        "top_holdings": [
            {"symbol": "Apple", "holdingPercent": "6.5%"},
            {"symbol": "Microsoft", "holdingPercent": "5.7%"},
            {"symbol": "Amazon", "holdingPercent": "4.3%"}
        ],
        "gastos": "0.0945%",
        "rango_1y": "360-420 USD",
        "rendimiento_ytd": "15%",
        "duracion": "Baja"
    },
    "VWO": {
        "nombre": "Vanguard FTSE Emerging Markets ETF",
        "descripcion": "Este ETF sigue el índice FTSE Emerging Markets All Cap China A Inclusion Index, que incluye acciones de mercados emergentes en Asia, Europa, América Latina y África.",
        "sector": "Renta variable",
        "categoria": "Acciones emergentes",
        "exposicion": "Mercados emergentes globales.",
        "moneda": "USD",
        "beta": 1.2,
        "top_holdings": [
            {"symbol": "Tencent", "holdingPercent": "6%"},
            {"symbol": "Alibaba", "holdingPercent": "4.5%"},
            {"symbol": "Taiwan Semiconductor", "holdingPercent": "4%"}
        ],
        "gastos": "0.08%",
        "rango_1y": "40-55 USD",
        "rendimiento_ytd": "10%",
        "duracion": "Alta"
    },
    "GLD": {
        "nombre": "SPDR Gold Shares",
        "descripcion": "Este ETF sigue el precio del oro físico.",
        "sector": "Materias primas",
        "categoria": "Oro físico",
        "exposicion": "Oro físico y contratos futuros de oro.",
        "moneda": "USD",
        "beta": 0.1,
        "top_holdings": [
            {"symbol": "Gold", "holdingPercent": "100%"}
        ],
        "gastos": "0.40%",
        "rango_1y": "160-200 USD",
        "rendimiento_ytd": "12%",
        "duracion": "Baja"
    }
}

# Función para cargar datos de múltiples tickers
def cargar_datos(tickers, inicio, fin):
    """
    Descarga datos históricos y calcula retornos diarios.
    """
    datos = {}
    for ticker in tickers:
        try:
            # Descargar datos
            df = yf.download(ticker, start=inicio, end=fin)
            # Procesar precios ajustados y calcular retornos
            df["Precio"] = df["Close"]
            df["Retornos"] = df["Close"].pct_change()
            datos[ticker] = df.dropna()
        except Exception as e:
            print(f"Error descargando datos para {ticker}: {e}")
    return datos

# Función para calcular beta
def calcular_beta(portfolio_returns, index_returns):
    cov_matrix = np.cov(portfolio_returns, index_returns)
    return cov_matrix[0, 1] / cov_matrix[1, 1]  # Covarianza / Varianza índice


# Función para calcular el ratio de Sharpe

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Asumiendo retornos diarios
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


# Función para calcular el ratio de Sortino

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation


# Función para calcular métricas estadísticas
def calcular_metricas(df, nivel_VaR=[0.95, 0.975, 0.99], risk_free_rate=0.02, target_return=0):
    """
    Calcula métricas estadísticas clave, incluyendo VaR, CVaR, Sharpe Ratio, Sortino, y otros.

    :param df: DataFrame con precios y retornos.
    :param nivel_VaR: Lista de niveles de confianza para VaR.
    :param risk_free_rate: Tasa libre de riesgo.
    :param target_return: Retorno objetivo.
    :return: Diccionario con métricas y serie de drawdown.
    """
    retornos = df['Retornos'].dropna()

    # Calcular métricas básicas
    media = np.mean(retornos) 
    volatilidad = np.std(retornos) 
    sesgo = skew(retornos)
    curtosis = kurtosis(retornos)

    # VaR y CVaR
    VaR = {f"VaR {nivel*100}%": (np.percentile(retornos, (1 - nivel) * 100))*100 for nivel in nivel_VaR}
    cVaR = {f"CVaR {nivel*100}%": (retornos[retornos <= np.percentile(retornos, (1 - nivel) * 100)].mean())*100 for nivel in nivel_VaR}

    # Ratios financieros
    sharpe = calcular_sharpe_ratio(retornos, risk_free_rate)
    sortino = calcular_sortino_ratio(retornos, risk_free_rate, target_return)

    # Crear diccionario con las métricas
    metrics = {
        "Media (%)": media*100,
        "Volatilidad": volatilidad,
        "Sesgo": sesgo,
        "Curtosis": curtosis,
        **VaR,
        **cVaR,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    }

    return metrics

# Función para cargar datos de múltiples tickers
def cargar_datos(tickers, inicio, fin):
    datos = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=inicio, end=fin)
            df["Precio"] = df['Close']
            df['Retornos'] = df['Close'].pct_change()
            datos[ticker] = df.dropna()  # Eliminamos filas con NaN
        except Exception as e:
            print(f"Error descargando datos para {ticker}: {e}")
    return datos

# Función para calcular drawdown
def calcular_drawdown(precios):
    high_water_mark = precios.expanding().max()
    drawdown = (precios - high_water_mark) / high_water_mark
    return drawdown, high_water_mark

# Función para graficar precios y drawdown
def graficar_drawdown_financiero(df, titulo="Análisis de Drawdown"):
    # Verificar que la columna 'Precios' esté presente
    if 'Precios' not in df.columns:
        df['Precios'] = df['Close']  # Usar 'Close' si 'Precios' no está

    # Calcular drawdown y high water mark
    drawdown, hwm = calcular_drawdown(df['Precios'])

    # Crear gráfico con dos subgráficas
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Graficar precios
    fig.add_trace(go.Scatter(x=df.index, y=df['Precios'].values, name='Precio', line=dict(color='blue')), row=1, col=1)
    # Graficar high water mark
    fig.add_trace(go.Scatter(x=hwm.index, y=hwm.values, name='High Water Mark', line=dict(color='green', dash='dash')), row=1, col=1)
    # Graficar drawdown
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name='Drawdown', line=dict(color='red'), fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'), row=2, col=1)

    # Actualizar layout y ejes
    fig.update_layout(
        title=titulo,
        height=800,
        showlegend=True,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_yaxes(title="Precio", row=1, col=1)
    fig.update_yaxes(title="Drawdown %", tickformat=".1%", range=[-1, 0.1], row=2, col=1)
    fig.update_xaxes(title="Fecha", row=2, col=1)

    return fig

# Función para graficar la distribución de rendimientos
def graficar_distribucion_rendimientos(rendimientos, titulo="Distribución de Rendimientos"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rendimientos, nbinsx=50, name="Rendimientos", marker=dict(color='blue', opacity=0.7)))
    fig.update_layout(
        title=titulo,
        xaxis_title="Rendimiento Diario",
        yaxis_title="Frecuencia",
        bargap=0.2,
        height=500
    )
    return fig

# Función para graficar rendimientos acumulados
def graficar_rendimientos_acumulados(df, titulo="Rendimientos Acumulados"):
    df['Rendimiento Acumulado'] = (1 + df['Retornos']).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Rendimiento Acumulado'], name="Rendimiento Acumulado", line=dict(color='blue')))
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Acumulado",
        height=500
    )
    return fig

# --- Funciones de Optimización de Portafolios ---
def optimizar_portafolio_markowitz(retornos, metodo="min_vol", objetivo=None):
    # Función para optimizar el portafolio según el modelo de Markowitz
    media = retornos.mean()
    cov = retornos.cov()

    # Función para calcular el riesgo del portafolio (volatilidad)
    def riesgo(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    # Función para calcular el Sharpe ratio del portafolio
    def sharpe(w):
        return -(np.dot(w.T, media) / np.sqrt(np.dot(w.T, np.dot(cov, w))))

    # Número de activos
    n = len(media)
    
    # Pesos iniciales (distribución igual)
    w_inicial = np.ones(n) / n
    
    # Restricciones: los pesos deben sumar 1
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if metodo == "target" and objetivo is not None:
        restricciones.append({"type": "eq", "fun": lambda w: np.dot(w, media) - objetivo})  # Rendimiento objetivo
        objetivo_funcion = riesgo
    elif metodo == "sharpe":
        objetivo_funcion = sharpe
    else:
        objetivo_funcion = riesgo
    
    # Definir los límites de los pesos (entre 0 y 1)
    limites = [(-1, 1) for _ in range(n)]
    
    # Optimización
    resultado = minimize(objetivo_funcion, w_inicial, constraints=restricciones, bounds=limites)
    
    # Devolver los pesos optimizados
    return np.array(resultado.x).flatten()


# --- Configuración de Streamlit ---
st.title("Proyecto de Optimización de Portafolios")

# Crear tabs
tabs = st.tabs(["Introducción", "Selección de ETF's", "Estadísticas de los ETF's", "Portafolios Óptimos", "Backtesting", "Modelo de Black-Litterman"])

# --- Introducción ---
with tabs[0]:
    st.header("Introducción")
    st.write(""" 
    Este proyecto tiene como objetivo analizar y optimizar un portafolio utilizando ETFs en diferentes clases de activos, tales como renta fija, renta variable, y materias primas. A lo largo del proyecto, se evaluará el rendimiento de estos activos a través de diversas métricas financieras y técnicas de optimización de portafolios, como la optimización de mínima volatilidad y la maximización del Sharpe Ratio. 
    """)

# --- Selección de ETF's ---
with tabs[1]:
    st.header("Selección de ETF's")
    hoy = "2024-12-2"
    datos_2010_hoy = cargar_datos(list(tickers.keys()), "2010-01-01", hoy)

    # Crear un DataFrame consolidado con las características de los ETFs
    etf_caracteristicas = pd.DataFrame({
        "Ticker": list(tickers.keys()),
        "Nombre": [info["nombre"] for info in tickers.values()],
        "Sector": [info["sector"] for info in tickers.values()],
        "Categoría": [info["categoria"] for info in tickers.values()],
        "Exposición": [info["exposicion"] for info in tickers.values()],
        "Moneda": [info["moneda"] for info in tickers.values()],
        "Beta": [info["beta"] for info in tickers.values()],
        "Gastos": [info["gastos"] for info in tickers.values()],
        "Rango 1 Año": [info["rango_1y"] for info in tickers.values()],
        "Rendimiento YTD": [info["rendimiento_ytd"] for info in tickers.values()],
        "Duración": [info["duracion"] for info in tickers.values()],
    })

    # Mostrar las características en Streamlit
    st.subheader("Características de los ETFs Seleccionados")
    st.dataframe(etf_caracteristicas)

    # Mostrar la serie de tiempo de cada ETF
    st.subheader("Series de Tiempo de los Precios de Cierre")
    for ticker, info in tickers.items():
        fig = px.line(datos_2010_hoy[ticker],
                      x=datos_2010_hoy[ticker].index,
                      y=datos_2010_hoy[ticker]['Close'].values.flatten(),
                      title=f"Precio de Cierre - {ticker}")
        st.plotly_chart(fig)
    
# --- Estadísticas de los ETF's ---
with tabs[2]:
    st.header("Estadísticas de los ETF's")
    # Configuración de fechas
    inicio = "2010-01-01"
    fin = "2023-01-01"

    # Descargar datos
    datos_etfs = cargar_datos(tickers, inicio, fin)

    # Inicializar diccionario para almacenar los rendimientos acumulados
    rendimientos_acumulados = {}

    # Ciclo para procesar cada ETF
    for ticker, data in datos_etfs.items():
        st.subheader(f"Análisis del ETF: {ticker}")
        
        # Calcular métricas
        metrics = calcular_metricas(data)
        
        # Mostrar métricas en una tabla
        st.markdown("### Métricas Estadísticas")
        st.table(pd.DataFrame(metrics, index=["Valor"]).T)

        # Graficar precios y drawdown
        fig = graficar_drawdown_financiero(data, titulo=f"Precios y Drawdown de {ticker}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Graficar distribución de rendimientos
        fig_dist = graficar_distribucion_rendimientos(data['Retornos'], titulo=f"Distribución de Rendimientos de {ticker}")
        st.plotly_chart(fig_dist, use_container_width=True)
    
        # Graficar rendimientos acumulados
        fig_acum = graficar_rendimientos_acumulados(data, titulo=f"Rendimientos Acumulados de {ticker}")
        st.plotly_chart(fig_acum, use_container_width=True)
        
        # Guardar los rendimientos acumulados de este ETF
        rendimientos_acumulados[ticker] = data['Rendimiento Acumulado']
    
    # Graficar todos los rendimientos acumulados al final
    st.subheader("Comparativa de Rendimientos Acumulados")
    fig_comparativa = go.Figure()
    for ticker, rend_acum in rendimientos_acumulados.items():
        fig_comparativa.add_trace(go.Scatter(x=rend_acum.index, y=rend_acum, name=ticker))
    
    fig_comparativa.update_layout(
        title="Rendimientos Acumulados de Todos los ETF's",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Acumulado",
        height=600
    )
    st.plotly_chart(fig_comparativa, use_container_width=True)


# --- Portafolios Óptimos ---
with tabs[3]:
    st.header("Portafolios Óptimos con la teoría de Markowitz")
        
    # Descargar datos históricos para el periodo 2010-2020
    datos_2010_2020 = cargar_datos(list(tickers.keys()), "2010-01-01", "2020-01-01")
    retornos_2010_2020 = pd.DataFrame({k: v["Retornos"] for k, v in datos_2010_2020.items()}).dropna()

    # 1. Portafolio de Mínima Volatilidad
    st.subheader("Portafolio de Mínima Volatilidad")
    pesos_min_vol = optimizar_portafolio_markowitz(tickers=list(tickers.keys()), datos=retornos_2010_2020, metodo="min_vol")
    st.write("Pesos del Portafolio de Mínima Volatilidad:")
    for ticker, peso in zip(tickers.keys(), pesos_min_vol):
        st.write(f"{ticker}: {peso:.2%}")
    fig_min_vol = px.bar(x=list(tickers.keys()), y=pesos_min_vol, title="Pesos - Mínima Volatilidad")
    st.plotly_chart(fig_min_vol)

    # 2. Portafolio de Máximo Sharpe Ratio
    st.subheader("Portafolio de Máximo Sharpe Ratio")
    pesos_sharpe = optimizar_portafolio_markowitz(tickers=list(tickers.keys()), datos=retornos_2010_2020, metodo="sharpe")
    st.write("Pesos del Portafolio de Máximo Sharpe Ratio:")
    for ticker, peso in zip(tickers.keys(), pesos_sharpe):
        st.write(f"{ticker}: {peso:.2%}")
    fig_sharpe = px.bar(x=list(tickers.keys()), y=pesos_sharpe, title="Pesos - Máximo Sharpe Ratio")
    st.plotly_chart(fig_sharpe)


    
# --- Backtesting ---
with tabs[4]:
    st.header("Backtesting")


    
# --- Modelo de Black-Litterman ---
with tabs[5]:
    st.header("Modelo de Black-Litterman")












