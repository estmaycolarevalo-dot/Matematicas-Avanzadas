"""
REPLICA BASE del metodo WPT-FFT de Aburakhia, Myers y Shami (2022)
Dataset: CWRU 12k Drive End.  Parametros del articulo: No=1200, k=3, m=1, db4, Random Forest.

Unica correccion respecto al codigo original de los autores:
  - la seleccion de la variable "_DE_time" dentro de cada .mat es EXPLICITA
    (el codigo original se quedaba con la ultima clave que coincidia, lo que
     funciona por casualidad en 99.mat, que trae duplicadas las variables de 98.mat).
"""
import glob, os, re, time, warnings
import numpy as np
import pywt
from scipy.io import loadmat
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

BASE = r"C:\Users\porta\dataset"
FS = 12000
NUM_SAMPLES = 1200      # No  -> T_vin = 0.1 s
K = 3                   # nivel de descomposicion WPT
M = 1                   # numero de frecuencias dominantes por forma de onda
WAVELET = 'db4'
SEED = 42

CLASES = [
    ('12K_DE_Normal',          1, 'Sana'),
    ('12k_DE_IRFault_0.007',   2, 'IR 0.007"'),
    ('12k_DE_IRFault_0.014',   3, 'IR 0.014"'),
    ('12k_DE_IRFault_0.021',   4, 'IR 0.021"'),
    ('12k_DE_BallFault_0.007', 5, 'Bola 0.007"'),
    ('12k_DE_BallFault_0.014', 6, 'Bola 0.014"'),
    ('12k_DE_BallFault_0.021', 7, 'Bola 0.021"'),
    ('12k_DE_ORFault_0.007',   8, 'OR 0.007"'),
    ('12k_DE_ORFault_0.014',   9, 'OR 0.014"'),
    ('12k_DE_ORFault_0.021',  10, 'OR 0.021"'),
]


# ---------- 1. Lectura y segmentacion ----------
def leer_senal_DE(ruta):
    """Devuelve la senal del acelerometro Drive End del archivo .mat.
    Elige la variable cuyo numero coincide con el nombre del archivo (ej: 99.mat -> X099_DE_time)."""
    datos = loadmat(ruta)
    num = os.path.splitext(os.path.basename(ruta))[0]
    esperada = 'X%03d_DE_time' % int(num)
    if esperada in datos:
        clave = esperada
    else:                                   # respaldo
        candidatas = [k for k in datos if re.search('_DE_time', k)]
        clave = candidatas[-1]
        print('   AVISO: %s no tiene %s, se usa %s' % (os.path.basename(ruta), esperada, clave))
    return datos[clave].reshape(-1), clave


def make_dataset(carpeta, num_samples, clase):
    X, Y = [], []
    for f in sorted(glob.glob(os.path.join(carpeta, '*.mat'))):
        senal, _ = leer_senal_DE(f)
        n_seg = len(senal) // num_samples
        X.append(senal[:n_seg * num_samples].reshape(n_seg, num_samples))
        Y.append(np.full(n_seg, clase))
    return np.concatenate(X), np.concatenate(Y)


# ---------- 2. Extraccion de caracteristicas WPT + FFT ----------
def apply_fft(x, fs, num_samples):
    f = np.linspace(0.0, fs / 2.0, num_samples // 2)
    v = fft(x)
    v = 2.0 / num_samples * np.abs(v[0:num_samples // 2])
    return f, v


def extraer_caracteristicas(X, k=K, m=M, wavelet=WAVELET, fs=FS):
    n_ondas = 2 ** k
    F = np.empty((len(X), m * n_ondas))
    for i in range(len(X)):
        wp = pywt.WaveletPacket(X[i], wavelet=wavelet, maxlevel=k)
        nodos = [n.path for n in wp.get_level(k, 'natural')]
        for j in range(n_ondas):
            nuevo = pywt.WaveletPacket(data=None, wavelet=wavelet, maxlevel=k)
            nuevo[nodos[j]] = wp[nodos[j]].data
            recon = nuevo.reconstruct(update=False)      # forma de onda elemental
            frec, amp = apply_fft(recon, fs, len(recon))
            idx = np.argpartition(amp, -m)[-m:]          # m amplitudes mas altas
            F[i, j * m:(j + 1) * m] = amp[idx] * frec[idx]   # Ec. (8)/(9)
    return F


# ---------- 3. Ejecucion ----------
if __name__ == '__main__':
    print('=' * 78)
    print('REPLICA BASE  |  No=%d (T_vin=%.3f s)  k=%d  m=%d  wavelet=%s  S=1x%d'
          % (NUM_SAMPLES, NUM_SAMPLES / FS, K, M, WAVELET, M * 2 ** K))
    print('=' * 78)

    Xs, Ys = [], []
    for carpeta, clase, nombre in CLASES:
        x, y = make_dataset(os.path.join(BASE, carpeta), NUM_SAMPLES, clase)
        Xs.append(x); Ys.append(y)
        print('  Clase %2d  %-12s %5d segmentos' % (clase, nombre, len(y)))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    print('  TOTAL: %d segmentos de %d puntos' % (X.shape[0], X.shape[1]))

    print('\nExtrayendo caracteristicas...')
    t0 = time.perf_counter()
    F = extraer_caracteristicas(X)
    t_extraccion = time.perf_counter() - t0
    print('  vector de caracteristicas: %s   (%.1f s en total)' % (str(F.shape), t_extraccion))

    Xtr, Xte, ytr, yte = train_test_split(F, Y, test_size=0.2, stratify=Y, random_state=SEED)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    clf = RandomForestClassifier(criterion='gini', max_features=1, min_samples_leaf=1,
                                 min_samples_split=2, max_depth=20, n_estimators=300,
                                 random_state=SEED)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    prob = clf.predict_proba(Xte_s)

    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, prob, multi_class='ovr')
    f1 = f1_score(yte, pred, average='micro')

    # tiempo de procesamiento en linea T_p de UN segmento
    t0 = time.perf_counter()
    for _ in range(50):
        f1seg = extraer_caracteristicas(X[:1])
        clf.predict(sc.transform(f1seg))
    Tp = (time.perf_counter() - t0) / 50
    Tvin = NUM_SAMPLES / FS

    print('\n' + '-' * 78)
    print('RESULTADOS')
    print('-' * 78)
    print('  Exactitud           = %.3f %%' % (acc * 100))
    print('  ROC AUC (ovr)       = %.3f' % auc)
    print('  F1 (micro)          = %.3f' % f1)
    print('  T_vin               = %.4f s' % Tvin)
    print('  T_p  (por segmento) = %.4f s' % Tp)
    print('  Retardo tau_d       = %.4f s     [Ec. (1)]' % (Tvin + Tp))

    print('\n  Exactitud por clase:')
    cm = confusion_matrix(yte, pred)
    for i, (_, clase, nombre) in enumerate(CLASES):
        print('    Clase %2d  %-12s  %3d/%3d  = %6.2f %%' %
              (clase, nombre, cm[i, i], cm[i].sum(), 100 * cm[i, i] / cm[i].sum()))

    print('\n  Matriz de confusion (filas = real, columnas = predicho):')
    print('      ' + ''.join('%5d' % c for _, c, _ in CLASES))
    for i, (_, clase, _) in enumerate(CLASES):
        print('  %3d ' % clase + ''.join('%5d' % v for v in cm[i]))

    np.save('F_base.npy', F); np.save('Y_base.npy', Y)
