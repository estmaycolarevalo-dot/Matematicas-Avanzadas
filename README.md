# Réplica y modificación del método híbrido WPT–FFT

Código de acompañamiento del informe *"Réplica y modificación de un método híbrido WPT–FFT para el monitoreo de condición de rodamientos: la transformada Z detrás del banco de filtros"* (Matemáticas Avanzadas).

Replica el método de Aburakhia, Myers y Shami (2022), *"A Hybrid Method for Condition Monitoring and Fault Diagnosis of Rolling Bearings With Low System Delay"* ([arXiv:2208.06051](https://arxiv.org/abs/2208.06051)), sobre el dataset [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), y luego implementa una parte del artículo que no estaba programada en el repositorio original de los autores.

## Dataset

Se necesitan 40 archivos `.mat` de la CWRU (36 de falla + 4 sanos, sección *12k Drive End Bearing Fault Data* y *Normal Baseline Data*), organizados en diez carpetas —una por clase— dentro de una ruta local. Esa ruta se configura editando la constante `BASE` al inicio de `base_wpt_fft.py`.

## Scripts

| Script | Qué hace |
|---|---|
| `base_wpt_fft.py` | Réplica base del método: lee y segmenta las señales, extrae 8 características por segmento (WPT de nivel k=3 + FFT, ecuación 9 del artículo) y clasifica con Random Forest. Corrige la lectura de la variable `_DE_time` dentro de cada `.mat` (antes dependía por casualidad del orden de las claves). Mide exactitud, AUC, F1, y el retardo del sistema τd = T_vin + T_p. |
| `modif_energia_entropia.py` | Modificación: implementa las ecuaciones (10)–(13) del artículo (razón energía/entropía para seleccionar wavelet base y nivel de descomposición), ausentes en el código de los autores. Corrige el signo faltante en la entropía de Shannon y usa el modo `periodization` para que la energía se conserve entre wavelets de distinta longitud. |
| `comparar.py` | Ejecuta el método completo con siete wavelets candidatas (usando las funciones de `base_wpt_fft.py`) para contrastar el ranking de `R(s)` contra la exactitud de clasificación realmente obtenida. |

## Orden de ejecución

```bash
pip install numpy scipy scikit-learn pywavelets

python base_wpt_fft.py            # réplica base (Tabla III/IV del informe)
python modif_energia_entropia.py  # criterio R(s) por wavelet y nivel (Tablas VI/VII)
python comparar.py                # exactitud real por wavelet (Tabla VIII)
```

## Nota

Estos scripts se apoyaron en herramientas de inteligencia artificial para completar partes del código publicado por los autores que no estaban implementadas (los datos, la optimización bayesiana no incluida, la construcción explícita del dataset, etc.). El análisis de resultados y las decisiones metodológicas están documentados en el informe.
