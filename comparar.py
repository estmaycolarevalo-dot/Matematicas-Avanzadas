import numpy as np, time, os, warnings
warnings.filterwarnings('ignore')
from base_wpt_fft import (make_dataset, extraer_caracteristicas, CLASES, BASE,
                          NUM_SAMPLES, FS, K, M, SEED)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

Xs,Ys=[],[]
for carpeta,clase,_ in CLASES:
    x,y=make_dataset(os.path.join(BASE,carpeta),NUM_SAMPLES,clase); Xs.append(x); Ys.append(y)
X=np.concatenate(Xs); Y=np.concatenate(Ys)

print('%-9s %6s %10s %8s %9s %9s %8s' % ('wavelet','R(s)','taps','Exact.%','AUC','T_p (s)','errores'))
print('-'*66)
Rs={'db2':20.470,'db4':20.925,'sym5':21.013,'db8':21.171,'coif3':21.260,'sym8':21.325,'coif5':21.351}
for w in ['db2','db4','sym5','db8','coif3','sym8','coif5']:
    t0=time.perf_counter(); F=extraer_caracteristicas(X,wavelet=w); tex=time.perf_counter()-t0
    Xtr,Xte,ytr,yte=train_test_split(F,Y,test_size=0.2,stratify=Y,random_state=SEED)
    sc=StandardScaler(); a=sc.fit_transform(Xtr); b=sc.transform(Xte)
    clf=RandomForestClassifier(criterion='gini',max_features=1,min_samples_leaf=1,min_samples_split=2,
                               max_depth=20,n_estimators=300,random_state=SEED).fit(a,ytr)
    p=clf.predict(b); pr=clf.predict_proba(b)
    acc=accuracy_score(yte,p); auc=roc_auc_score(yte,pr,multi_class='ovr')
    err=(p!=yte).sum()
    import pywt; taps=len(pywt.Wavelet(w).dec_lo)
    print('%-9s %6.2f %10d %8.3f %9.4f %9.5f %8d' % (w,Rs[w],taps,acc*100,auc,tex/len(X),err))
    np.save('F_%s.npy'%w,F)
np.save('Y_all.npy',Y)
