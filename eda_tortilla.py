import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RUTA_CSV = "tortilla_prices.csv"
CARPETA_FIG = "figuras"
YEAR_REF = 2025
os.makedirs(CARPETA_FIG, exist_ok=True)

COL = {"Tortillerías": "#D55E00", "Autoservicio": "#0072B2"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
FUENTE = "Fuente: SNIIM, Secretaría de Economía (vía Kaggle, R. Chavelas)"

df = pd.read_csv(RUTA_CSV)
n_inicial = len(df)

df["State"] = (df["State"].str.replace("\xa0", " ", regex=False).str.strip()
               .replace({"D.F.": "Ciudad de México", "Edo. México": "Estado de México"}))
df["City"] = df["City"].str.replace("\xa0", " ", regex=False).str.strip()

df["Canal"] = df["Store type"].map({"Mom and Pop Store": "Tortillerías",
                                    "Big Retail Store": "Autoservicio"})
df["Fecha"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))

n_faltantes = df["Price per kilogram"].isna().sum()
n_ceros = (df["Price per kilogram"] <= 0).sum()

df["Precio"] = df["Price per kilogram"].where(df["Price per kilogram"] > 0)
d = df.dropna(subset=["Precio"]).copy()

print(f"Registros: {n_inicial:,} | faltantes: {n_faltantes:,} | precios <= 0: {n_ceros}")
print(f"Rango de fechas: {df.Fecha.min().date()} a {df.Fecha.max().date()}")
print("Faltantes por año:\n", df[df['Price per kilogram'].isna()].groupby('Year').size().to_string())

#1. Univariada
d25 = d[d.Year == YEAR_REF]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(d25["Precio"], bins=30, color="#7f7f7f", edgecolor="white")
ax.axvline(d25["Precio"].median(), color="black", ls="--", lw=1)
ax.text(d25["Precio"].median(), ax.get_ylim()[1]*0.95,
        f" Mediana: ${d25['Precio'].median():.1f}", va="top")
ax.set_title(f"Distribución del precio por kilo de tortilla, {YEAR_REF}")
ax.set_xlabel("Precio por kg")
ax.set_ylabel("Número de registros (ciudad-día-canal)")
fig.text(0.01, 0.01, FUENTE, fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{CARPETA_FIG}/fig1_histograma_precio_{YEAR_REF}.png"); plt.close(fig)

#2. Bivariada
est = (d25.groupby("State")["Precio"].mean().sort_values())
n_ciu = d25.groupby("State")["City"].nunique()
fig, ax = plt.subplots(figsize=(7, 8))
colores = ["#7f7f7f"] * len(est)
colores[0] = colores[-1] = "#0072B2"          # resalta mínimo y máximo
ax.barh(est.index, est.values, color=colores)
ax.axvline(d25["Precio"].mean(), color="black", ls="--", lw=1)
for i, (s, v) in enumerate(est.items()):
    ax.text(v + 0.2, i, f"${v:.1f}  ({n_ciu[s]} ciudad{'es' if n_ciu[s] > 1 else ''})",
            va="center", fontsize=7)
ax.set_xlim(0, est.max() * 1.25)               
ax.set_title(f"Precio promedio por kilo de tortilla por entidad, {YEAR_REF}")
ax.set_xlabel("Precio por kg")
fig.text(0.01, 0.005, FUENTE, fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(f"{CARPETA_FIG}/fig2_barras_entidad_{YEAR_REF}.png"); plt.close(fig)

#3. Multivariada
h = (d[d.Year <= YEAR_REF].groupby(["State", "Year", "Canal"])["Precio"].mean().reset_index())
orden = (h[h.Year == YEAR_REF].groupby("State")["Precio"].mean().sort_values(ascending=False).index)
vmin, vmax = h["Precio"].min(), h["Precio"].max()
fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
for ax, canal in zip(axes, ["Tortillerías", "Autoservicio"]):
    m = h[h.Canal == canal].pivot(index="State", columns="Year", values="Precio").reindex(orden)
    sns.heatmap(m, ax=ax, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                cbar=(canal == "Autoservicio"), cbar_kws={"label": "Precio por kg"},
                linewidths=0.3, linecolor="white")
    ax.set_title(canal); ax.set_xlabel("Año"); ax.set_ylabel("")
fig.suptitle(f"Precio promedio por kilo por entidad, año y canal de venta, 2007-{YEAR_REF}", y=0.99)
fig.text(0.01, 0.005, FUENTE + " Entidades ordenadas por precio en 2025. Mismo color = mismo valor en ambos paneles.",
         fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(f"{CARPETA_FIG}/fig3_mapa_calor_entidad_canal.png"); plt.close(fig)

#4. Temporal
t = (d.set_index("Fecha").groupby("Canal")["Precio"].resample("MS").mean().reset_index())
fig, ax = plt.subplots(figsize=(9, 4.5))
for canal, g in t.groupby("Canal"):
    ax.plot(g["Fecha"], g["Precio"], color=COL[canal], lw=2, label=canal)
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color="gray", alpha=0.15)
ax.text(pd.Timestamp("2020-07-01"), 1, "2020: datos faltantes", ha="center", fontsize=8, color="dimgray")
ax.set_ylim(0, None)
ax.set_title("Precio promedio mensual por kilo de tortilla según canal de venta, 2007-2026")
ax.set_ylabel("Precio por kg"); ax.legend(frameon=False)
fig.text(0.01, 0.01, FUENTE + " 2026: enero a abril.", fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{CARPETA_FIG}/fig4_serie_tiempo_canal.png"); plt.close(fig)

#5. Comparación de grupos
years = [2007, 2013, 2019, 2025]
b = d[d.Year.isin(years)]
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.boxplot(data=b, x="Year", y="Precio", hue="Canal", hue_order=list(COL), palette=COL,
            fliersize=1.5, linewidth=0.9, ax=ax)
ax.set_ylim(0, None)
ax.set_title("Distribución del precio por kilo según canal de venta")
ax.set_xlabel("Año"); ax.set_ylabel("Precio por kg")
ax.legend(title="", frameon=False)
fig.text(0.01, 0.01, FUENTE, fontsize=7, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{CARPETA_FIG}/fig5_boxplot_canal_year.png"); plt.close(fig)

print("\nPromedio anual por canal:")
print(d[d.Year.isin(years + [2026])].groupby(["Year", "Canal"])["Precio"].mean().unstack().round(2))
print(f"\nEntidades más baratas / caras {YEAR_REF}:")
print(est.head(3).round(2).to_string()); print(est.tail(3).round(2).to_string())
print("\nMediana por canal 2025:", d25.groupby("Canal")["Precio"].median().round(2).to_dict())
