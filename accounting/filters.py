# src/accounting/filters.py
from typing import Callable, Dict
import pandas as pd

def by_tipo(tipo: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df["Tipo"].fillna("") == tipo

def by_flujo_contains(substr: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df["Flujo"].fillna("").str.contains(substr, na=False)

def cobros_renta() -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: (df["Flujo"] == "Cobros") & (df["Tipo"] == "Renta")

# Registry — add new rules here
FILTERS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "Impuestos": by_tipo("Impuestos"),
    "Servicio": by_tipo("Servicio"),
    "Legal": by_tipo("Legal"),
    "Cambio_FX": by_tipo("FX"),
    "Contrib_Inquilinos": by_flujo_contains("Contrib Inq"),
    "Contribuciones": lambda df: df["Flujo"] == "Contribucion",
    "Contrib_Inquilinos_Caja": lambda df: df["Flujo"] == "Contrib Inq Caja",
    "Contrib_Inquilinos_Directo": lambda df: df["Flujo"] == "Contrib Inq Dir",
    "Contrib_Impuestos": lambda df: (df["Flujo"] == "Contribucion") & (df["Tipo"] == "Impuestos"),
    "Cobros_Renta": cobros_renta(),
    "Pagos": lambda df: df["Flujo"] == "Pagos",
    "Repago": lambda df: df["Flujo"] == "Repago",
    "Transfer": lambda df: df["Flujo"] == "Transfer",
}





    # # Define masks for custom filtering logic
    # masks = {
    #     # "Cambio_FX": (df["Flujo"] == "Cambio") & (df["Tipo"] == "FX"),

    #     "Cobros_Renta": (df["Flujo"] == "Cobros") & (df["Tipo"] == "Renta"),


    #     "Impuestos": (df["Tipo"] == "Impuestos"),
    #     "Servicio": (df["Tipo"] == "Servicio"),
    #     "Legal": (df["Tipo"] == "Legal"),
    #     "Cambio_FX": (df["Tipo"] == "FX"),

    #     "Contrib_Inquilinos": df["Flujo"].str.contains("Contrib Inq"),
    #     "Contribuciones": df["Flujo"] == "Contribucion",

    #     # Specific combinations for Contribuciones and Pagos
    #     "Contrib_Inquilinos_Caja": (df["Flujo"] == "Contrib Inq Caja"), # & (df["Tipo"] == "Impuestos"),
    #     "Contrib_Inquilinos_Directo": (df["Flujo"] == "Contrib Inq Dir"), # # & (df["Tipo"] == "Impuestos"),
    #     "Contrib_Impuestos": (df["Flujo"] == "Contribucion") & (df["Tipo"] == "Impuestos"),
    #     "Contrib_Costos_Legales": (df["Flujo"] == "Contribucion") & (df["Tipo"] == "Legal"),
    #     "Contrib_Servicio": (df["Flujo"] == "Contribucion") & (df["Tipo"] == "Servicio"),
    #     "Costo_Operativo_Medio": (df["Flujo"] == "Costo Operativo Medio") & (df["Tipo"] == "Impuestos"),
    #     "Dividendos": (df["Flujo"] == "Dividendos") & (df["Tipo"] == "Dividendo"),


    #     "Pagos": (df["Flujo"] == "Pagos"),
    #     "Pagos_Impuestos": (df["Flujo"] == "Pagos") & (df["Tipo"] == "Impuestos"),
    #     "Pagos_Legal": (df["Flujo"] == "Pagos") & (df["Tipo"] == "Legal"),
        
    #     "Repago": (df["Flujo"] == "Repago"),

    #     "Transfer_Transfer": (df["Flujo"] == "Transfer"),
    # }
