"""
MODIFICACION 1 — Implementacion de las Ecuaciones (10) a (13) del articulo
(seleccion de la wavelet base y del nivel de descomposicion por la razon energia/entropia)

Estas ecuaciones estan descritas en la Seccion IV-A del articulo pero NO estan
programadas en el repositorio de los autores. Aqui se implementan y se usan para
verificar si db4 y k=3 eran realmente las mejores elecciones.

NOTA SOBRE LA Ec. (11): tal como esta impresa en el articulo,
      Entropia_i = SUM p_i * log2(p_i)
le falta el signo menos de la definicion de Shannon. Tal cual daria siempre un
valor negativo y R(s) = E/Entropia seria negativo, invirtiendo el criterio
"a mayor R(s), mejor". Aqui se usa la formula correcta:
      Entropia_i = - SUM p_i * log2(p_i)
"""
import numpy as np, pywt, warnings
from base_wpt_fft import make_dataset, CLASES, BASE, NUM_SAMPLES, FS
import os
warnings.filterwarnings('ignore')

SEED = 42
N_MUESTRA = 1000          # segmentos usados para evaluar R(s)
CANDIDATAS = ['db2', 'db4', 'db6', 'db8', 'db10', 'sym4', 'sym5', 'sym8', 'coif2', 'coif3', 'coif5']
NIVELES = [2, 3, 5]


def energia_entropia(x, wavelet, k):
    """Ecs. (10), (11), (12) y (13) aplicadas a un segmento."""
    wp = pywt.WaveletPacket(x, wavelet=wavelet, mode='periodization', maxlevel=k)
    w = np.concatenate([n.data for n in wp.get_level(k, 'natural')])   # coeficientes del nivel k
    E = np.sum(np.abs(w) ** 2)                                         # Ec. (10)
    p = np.abs(w) ** 2 / E                                             # Ec. (12)
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))                                        # Ec. (11) corregida
    return E, H, E / H                                                 # Ec. (13)


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print('Cargando una muestra de segmentos de las 10 clases...')
    Xs = []
    for carpeta, clase, _ in CLASES:
        x, _ = make_dataset(os.path.join(BASE, carpeta), NUM_SAMPLES, clase)
        Xs.append(x[rng.choice(len(x), N_MUESTRA // 10, replace=False)])
    X = np.concatenate(Xs)
    print('  %d segmentos de %d puntos\n' % (X.shape[0], X.shape[1]))

    print('=' * 84)
    print('RAZON ENERGIA/ENTROPIA  R(s) = E_i / Entropia_i    [Ec. (13)]')
    print('  (promedio sobre %d segmentos; a mayor R(s), mejor la wavelet/nivel)' % len(X))
    print('=' * 84)
    print('%-9s %-7s %10s %10s %12s %10s' % ('wavelet', 'taps', 'k', 'Energia E', 'Entropia', 'R(s)'))
    print('-' * 84)

    resultados = {}
    for wname in CANDIDATAS:
        taps = len(pywt.Wavelet(wname).dec_lo)
        for k in NIVELES:
            vals = np.array([energia_entropia(x, wname, k) for x in X])
            E, H, R = vals.mean(0)
            resultados[(wname, k)] = (E, H, R)
            print('%-9s %-7d %10d %10.2f %12.4f %10.3f' % (wname, taps, k, E, H, R))
        print('-' * 84)

    print('\nRANKING para k = 3 (el nivel que usa el articulo):')
    r3 = sorted([(v[2], w) for (w, k), v in resultados.items() if k == 3], reverse=True)
    for i, (R, w) in enumerate(r3, 1):
        marca = '   <-- la que usa el articulo' if w == 'db4' else ''
        print('  %2d.  %-8s R(s) = %8.3f%s' % (i, w, R, marca))

    print('\nRANKING de niveles k (con la mejor wavelet, %s):' % r3[0][1])
    mejor = r3[0][1]
    for k in NIVELES:
        print('  k = %d  ->  R(s) = %8.3f' % (k, resultados[(mejor, k)][2]))

    np.save('R_resultados.npy', resultados, allow_pickle=True)
