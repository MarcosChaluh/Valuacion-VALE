"""
Modelo modular de valuación DCF + múltiplos para VALE utilizando los Excel cargados
en el repositorio. El script está organizado en secciones:
1. Utilidades de lectura de Excel y mapeo de columnas.
2. Cálculo de FCF y supuestos operativos.
3. Cálculo de WACC.
4. Proyección de flujos y valoración (EV, equity y valor por acción).
5. Valuación relativa por múltiplos.

El archivo `summary-30-09-2025.xls` se usa como fuente automática de métricas
históricas (Revenue, EBITDA, EBIT, Net Income, Capex, Working Capital, Deuda y
Caja) para que el modelo arranque directamente con los datos reales incluidos en
el proyecto.

Requiere las librerías pandas y numpy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Lectura y preparación de datos
# ---------------------------

def listar_estructura_excel(ruta: str) -> Dict[str, List[str]]:
    """Devuelve las hojas y las primeras columnas de cada hoja para guiar el mapeo."""
    libro = pd.ExcelFile(ruta)
    estructura = {}
    for hoja in libro.sheet_names:
        df = libro.parse(hoja, nrows=5)
        estructura[hoja] = [str(c) for c in df.columns]
    return estructura


def cargar_summary_financiero(ruta: str, periodo: str = "2024") -> Dict[str, float]:
    """
    Lee el archivo de resumen y devuelve métricas clave del período solicitado.

    Espera columnas de año (por ejemplo, "2024" o "TTM") y filas con nombres
    estándar como "Revenue", "EBITDA", "EBIT", "Net Income", "Capital Expenditure",
    "Working Capital", "Total Debt", "Cash And Cash Equivalents", "Shares Outstanding
    Capital".
    """

    df = pd.read_excel(ruta, sheet_name=0)
    df.rename(columns={df.columns[0]: "metric"}, inplace=True)
    df.set_index("metric", inplace=True)

    def _extraer_valor(nombre: str) -> float:
        if nombre not in df.index:
            raise KeyError(f"No se encontró la fila '{nombre}' en {ruta}")
        valor = df.loc[nombre, periodo]
        if pd.isna(valor):
            raise ValueError(f"La fila '{nombre}' no tiene datos para {periodo}")
        return float(valor)

    def _normalizar_acciones(valor: float) -> float:
        """Aplica un factor si la cifra luce inflada (e.g., 1e18)."""

        if valor <= 0:
            return np.nan
        candidatos = [valor, valor / 1e3, valor / 1e6, valor / 1e9]
        for cand in candidatos:
            if 1e8 <= cand <= 1e11:  # rango razonable de acciones en circulación
                return cand
        # si no hay candidato razonable, devolver sin escalar
        return valor

    def _extraer_opcional(nombre: str) -> float:
        try:
            return _extraer_valor(nombre)
        except Exception:
            return np.nan

    metrics = {
        "revenue": _extraer_valor("Revenue"),
        "ebitda": _extraer_valor("EBITDA"),
        "ebit": _extraer_valor("EBIT"),
        "net_income": _extraer_valor("Net Income"),
        "capex": _extraer_valor("Capital Expenditure"),
        "working_capital": _extraer_valor("Working Capital"),
        "total_debt": _extraer_valor("Total Debt"),
        "cash": _extraer_valor("Cash And Cash Equivalents"),
        "equity": _extraer_valor("Total Equity"),
        "shares_outstanding": _normalizar_acciones(_extraer_opcional("Shares Outstanding Capital")),
        "book_value_per_share": _extraer_opcional("Book Value per Share"),
    }

    # Para la variación de capital de trabajo usamos el año previo si existe
    anos_disponibles = [c for c in df.columns if str(c).isdigit()]
    if periodo.isdigit():
        prev_ano = str(int(periodo) - 1)
        if prev_ano in anos_disponibles and "Working Capital" in df.index:
            wc_prev = df.loc["Working Capital", prev_ano]
            metrics["delta_wc"] = metrics["working_capital"] - float(wc_prev)
        else:
            metrics["delta_wc"] = 0.0
    else:
        metrics["delta_wc"] = 0.0

    # Depreciación se extrae del cash flow detallado para mejorar consistencia
    try:
        df_cf = pd.read_excel("Cash Flow_Annual_Restated.xls", sheet_name=0)
        df_cf.rename(columns={df_cf.columns[0]: "metric"}, inplace=True)
        df_cf.set_index("metric", inplace=True)
        metrics["depreciation"] = float(df_cf.loc["Depreciation, Amortization and Depletion", periodo])
    except Exception:
        metrics["depreciation"] = metrics["ebitda"] - metrics["ebit"]

    if np.isnan(metrics["shares_outstanding"]) and metrics.get("book_value_per_share") and metrics["book_value_per_share"] > 0:
        metrics["shares_outstanding"] = metrics["equity"] / metrics["book_value_per_share"]

    metrics["net_debt"] = metrics["total_debt"] - metrics["cash"]
    return metrics

def mapear_columnas(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Mapea nombres estándar a columnas reales usando alias de keywords.
    aliases espera {'ventas': ['Revenue', ...], 'ebit': ['EBIT', ...], etc.}
    """
    columnas = {c.lower(): c for c in df.columns}
    mapeo: Dict[str, str] = {}
    for clave, candidatos in aliases.items():
        for cand in candidatos:
            cand_low = cand.lower()
            for col_low, col_real in columnas.items():
                if cand_low in col_low:
                    mapeo[clave] = col_real
                    break
            if clave in mapeo:
                break
    return mapeo


# ---------------------------
# Supuestos y WACC
# ---------------------------

@dataclass
class SupuestosOperativos:
    crecimiento_ventas: List[float]
    margen_ebit: List[float]
    capex_sobre_ventas: List[float]
    depreciacion_sobre_ventas: List[float]
    capital_trabajo_sobre_ventas: List[float]
    tasa_impuesto: float


@dataclass
class EntradasWACC:
    tasa_libre_riesgo: float
    prima_riesgo_mercado: float
    prima_riesgo_pais: float
    beta_apalancada: float
    costo_deuda_pre_impuesto: float
    tasa_impuesto: float
    deuda_sobre_equity_objetivo: float


@dataclass
class ResultadosWACC:
    costo_equity: float
    costo_deuda_despues_impuestos: float
    wacc: float


def calcular_costo_equity(rf: float, prm: float, prp: float, beta: float) -> float:
    return rf + beta * (prm + prp)


def calcular_costo_deuda_despues_impuestos(costo_pre_tax: float, tasa_impuesto: float) -> float:
    return costo_pre_tax * (1 - tasa_impuesto)


def calcular_wacc(entradas: EntradasWACC) -> ResultadosWACC:
    costo_equity = calcular_costo_equity(
        entradas.tasa_libre_riesgo,
        entradas.prima_riesgo_mercado,
        entradas.prima_riesgo_pais,
        entradas.beta_apalancada,
    )
    costo_deuda = calcular_costo_deuda_despues_impuestos(
        entradas.costo_deuda_pre_impuesto, entradas.tasa_impuesto
    )
    d_over_e = entradas.deuda_sobre_equity_objetivo
    peso_deuda = d_over_e / (1 + d_over_e)
    peso_equity = 1 - peso_deuda
    wacc = peso_equity * costo_equity + peso_deuda * costo_deuda
    return ResultadosWACC(costo_equity, costo_deuda, wacc)


# ---------------------------
# FCF y proyecciones
# ---------------------------

def calcular_fcf(ebit: float, tasa_impuesto: float, depreciacion: float, capex: float, variacion_ct: float) -> float:
    nopat = ebit * (1 - tasa_impuesto)
    return nopat + depreciacion - capex - variacion_ct


def proyectar_fcf(
    ventas_iniciales: float,
    supuestos: SupuestosOperativos,
) -> pd.DataFrame:
    """
    Proyecta ventas, EBIT, depreciación, capex, capital de trabajo y FCF.
    Listas en supuestos deben tener longitud igual al número de años explícitos.
    """
    anos = len(supuestos.crecimiento_ventas)
    registros = []
    ventas = ventas_iniciales
    for t in range(anos):
        crecimiento = supuestos.crecimiento_ventas[t]
        ventas = ventas * (1 + crecimiento)
        margen_ebit = supuestos.margen_ebit[t]
        ebit = ventas * margen_ebit
        dep = ventas * supuestos.depreciacion_sobre_ventas[t]
        capex = ventas * supuestos.capex_sobre_ventas[t]
        capital_trabajo = ventas * supuestos.capital_trabajo_sobre_ventas[t]
        if registros:
            variacion_ct = capital_trabajo - registros[-1]["capital_trabajo"]
        else:
            variacion_ct = capital_trabajo
        fcf = calcular_fcf(ebit, supuestos.tasa_impuesto, dep, capex, variacion_ct)
        registros.append(
            {
                "ano": t + 1,
                "ventas": ventas,
                "ebit": ebit,
                "depreciacion": dep,
                "capex": capex,
                "capital_trabajo": capital_trabajo,
                "variacion_ct": variacion_ct,
                "fcf": fcf,
            }
        )
    return pd.DataFrame(registros)


def valor_terminal_gordon(fcf_ultimo: float, wacc: float, g: float) -> float:
    return fcf_ultimo * (1 + g) / (wacc - g)


def valor_terminal_multiple(ebitda_salida: float, multiple_ev_ebitda: float) -> float:
    return ebitda_salida * multiple_ev_ebitda


def descontar_flujos(flujos: pd.Series, wacc: float) -> pd.Series:
    anos = np.arange(1, len(flujos) + 1)
    factores = 1 / (1 + wacc) ** anos
    return flujos * factores


def valuar_empresa(
    df_proyeccion: pd.DataFrame,
    wacc: float,
    crecimiento_terminal: Optional[float] = None,
    multiple_salida: Optional[float] = None,
    ebitda_salida: Optional[float] = None,
    deuda_neta: float = 0.0,
    acciones: float = 1.0,
) -> Dict[str, float]:
    """
    Calcula EV, equity y valor por acción con TV por Gordon o múltiplo.
    """
    flujos = df_proyeccion["fcf"]
    flujos_desc = descontar_flujos(flujos, wacc)
    valor_actual_flujos = flujos_desc.sum()

    if crecimiento_terminal is not None:
        tv = valor_terminal_gordon(flujos.iloc[-1], wacc, crecimiento_terminal)
    elif multiple_salida is not None and ebitda_salida is not None:
        tv = valor_terminal_multiple(ebitda_salida, multiple_salida)
    else:
        raise ValueError("Debe definirse crecimiento_terminal o múltiplo de salida")

    tv_desc = tv / (1 + wacc) ** len(flujos)
    ev = valor_actual_flujos + tv_desc
    equity = ev - deuda_neta
    valor_accion = equity / acciones if acciones else np.nan
    return {
        "valor_actual_flujos": valor_actual_flujos,
        "valor_terminal": tv,
        "valor_terminal_descontado": tv_desc,
        "enterprise_value": ev,
        "equity_value": equity,
        "valor_por_accion": valor_accion,
    }


# ---------------------------
# Valuación relativa
# ---------------------------

def calcular_multiplos(comparables: pd.DataFrame) -> pd.DataFrame:
    """Espera columnas: EV, EquityValue, EBITDA, NetIncome, BookValue, Revenue."""
    df = comparables.copy()
    df["EV_EBITDA"] = df["EV"] / df["EBITDA"]
    df["P_E"] = df["EquityValue"] / df["NetIncome"]
    df["P_BV"] = df["EquityValue"] / df["BookValue"]
    df["EV_Sales"] = df["EV"] / df["Revenue"]
    return df


def valuacion_implicita_por_multiplos(multiplos: pd.DataFrame, metricas_objetivo: Dict[str, float]) -> pd.DataFrame:
    """
    Usa los múltiplos promedio de comparables para derivar valores implícitos.
    metricas_objetivo debe incluir: EBITDA, NetIncome, BookValue, Revenue, NetDebt, Shares.
    """
    promedios = multiplos[["EV_EBITDA", "P_E", "P_BV", "EV_Sales"]].mean()
    resultados = {
        "EV_por_EBITDA": promedios["EV_EBITDA"] * metricas_objetivo["EBITDA"],
        "Equity_por_PE": promedios["P_E"] * metricas_objetivo["NetIncome"],
        "Equity_por_PBV": promedios["P_BV"] * metricas_objetivo["BookValue"],
        "EV_por_Sales": promedios["EV_Sales"] * metricas_objetivo["Revenue"],
    }
    equity_por_ev_prom = (resultados["EV_por_EBITDA"] + resultados["EV_por_Sales"]) / 2 - metricas_objetivo[
        "NetDebt"
    ]
    valores_por_accion = {
        "Valor_accion_PE": resultados["Equity_por_PE"] / metricas_objetivo["Shares"],
        "Valor_accion_PBV": resultados["Equity_por_PBV"] / metricas_objetivo["Shares"],
        "Valor_accion_promedio_EV": equity_por_ev_prom / metricas_objetivo["Shares"],
    }
    return pd.DataFrame({"valuation": list(resultados.keys()) + list(valores_por_accion.keys()),
                         "valor": list(resultados.values()) + list(valores_por_accion.values())})


# ---------------------------
# Ejemplo de uso y supuestos editables
# ---------------------------

def ejemplo_configuracion() -> Tuple[SupuestosOperativos, EntradasWACC]:
    supuestos = SupuestosOperativos(
        crecimiento_ventas=[0.04, 0.03, 0.03, 0.025, 0.02],
        margen_ebit=[0.23, 0.24, 0.24, 0.24, 0.24],
        capex_sobre_ventas=[0.08, 0.08, 0.08, 0.08, 0.08],
        depreciacion_sobre_ventas=[0.05, 0.05, 0.05, 0.05, 0.05],
        capital_trabajo_sobre_ventas=[0.10, 0.10, 0.10, 0.10, 0.10],
        tasa_impuesto=0.27,
    )
    entradas_wacc = EntradasWACC(
        tasa_libre_riesgo=0.045,
        prima_riesgo_mercado=0.055,
        prima_riesgo_pais=0.02,
        beta_apalancada=1.1,
        costo_deuda_pre_impuesto=0.065,
        tasa_impuesto=supuestos.tasa_impuesto,
        deuda_sobre_equity_objetivo=0.4,
    )
    return supuestos, entradas_wacc


def supuestos_desde_datos(metrics: Dict[str, float], anos: int = 5) -> SupuestosOperativos:
    """Construye supuestos base a partir de ratios históricos de los Excel."""

    margen_ebit_hist = metrics["ebit"] / metrics["revenue"] if metrics["revenue"] else 0.2
    dep_ratio = metrics.get("depreciation", 0.0) / metrics["revenue"] if metrics["revenue"] else 0.05
    capex_ratio = metrics["capex"] / metrics["revenue"] if metrics["revenue"] else 0.08
    ct_ratio = metrics["working_capital"] / metrics["revenue"] if metrics["revenue"] else 0.05

    crecimiento_base = 0.02
    crecimiento = [crecimiento_base for _ in range(anos)]
    margen = [margen_ebit_hist for _ in range(anos)]
    capex_r = [capex_ratio for _ in range(anos)]
    dep_r = [dep_ratio for _ in range(anos)]
    ct_r = [ct_ratio for _ in range(anos)]

    return SupuestosOperativos(
        crecimiento_ventas=crecimiento,
        margen_ebit=margen,
        capex_sobre_ventas=capex_r,
        depreciacion_sobre_ventas=dep_r,
        capital_trabajo_sobre_ventas=ct_r,
        tasa_impuesto=0.27,
    )


if __name__ == "__main__":
    periodo_base = "2024"  # puede cambiarse a "TTM" para trailing 12 meses

    print("\nLectura de archivos de estados financieros cargados en el proyecto...")
    estructura_summary = listar_estructura_excel("summary-30-09-2025.xls")
    print("Estructura de summary-30-09-2025.xls:")
    for hoja, columnas in estructura_summary.items():
        print(f"- {hoja}: {columnas}")

    metrics = cargar_summary_financiero("summary-30-09-2025.xls", periodo=periodo_base)
    print("\nMétricas extraídas del summary:")
    for k, v in metrics.items():
        print(f"{k}: {v:,.2f}")

    supuestos = supuestos_desde_datos(metrics)
    entradas_wacc = EntradasWACC(
        tasa_libre_riesgo=0.045,
        prima_riesgo_mercado=0.055,
        prima_riesgo_pais=0.02,
        beta_apalancada=1.1,
        costo_deuda_pre_impuesto=0.065,
        tasa_impuesto=supuestos.tasa_impuesto,
        deuda_sobre_equity_objetivo=metrics["net_debt"] / metrics["equity"] if metrics["equity"] else 0.4,
    )
    res_wacc = calcular_wacc(entradas_wacc)
    print("\nCosto de equity: {:.2%}, Costo de deuda after tax: {:.2%}, WACC: {:.2%}".format(
        res_wacc.costo_equity, res_wacc.costo_deuda_despues_impuestos, res_wacc.wacc
    ))

    df_proj = proyectar_fcf(metrics["revenue"], supuestos)
    print("\nProyección de FCF (USD):")
    print(df_proj)

    resultados = valuar_empresa(
        df_proj,
        wacc=res_wacc.wacc,
        crecimiento_terminal=0.02,
        deuda_neta=metrics["net_debt"],
        acciones=metrics["shares_outstanding"] if not np.isnan(metrics["shares_outstanding"]) else 1.0,
    )
    print("\nResultados de valuación DCF:")
    for k, v in resultados.items():
        print(f"{k}: {v:,.2f}")

    comparables = pd.DataFrame(
        {
            "EV": [60000, 52000, 48000],
            "EquityValue": [42000, 36000, 33000],
            "EBITDA": [10000, 9000, 8500],
            "NetIncome": [5500, 4800, 4600],
            "BookValue": [30000, 28000, 27000],
            "Revenue": [45000, 43000, 41000],
        }
    )
    multiplos = calcular_multiplos(comparables)
    metricas_obj = {
        "EBITDA": metrics["ebitda"],
        "NetIncome": metrics["net_income"],
        "BookValue": metrics["equity"],
        "Revenue": metrics["revenue"],
        "NetDebt": metrics["net_debt"],
        "Shares": metrics["shares_outstanding"] if not np.isnan(metrics["shares_outstanding"]) else 1.0,
    }
    valuaciones_rel = valuacion_implicita_por_multiplos(multiplos, metricas_obj)
    print("\nValuación relativa (implícita):")
    print(valuaciones_rel)

    print("\nAjuste los supuestos en 'supuestos_desde_datos' o reemplace periodo_base para recalcular con TTM.")
