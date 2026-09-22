from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve
X,y=load_breast_cancer(return_X_y=True,as_frame=True)
# Construct a controlled rare-positive study from real observations only: keep all class 0 and a seeded subset of class 1.
rng=np.random.default_rng(42); idx0=np.where(y.to_numpy()==0)[0]; idx1=np.where(y.to_numpy()==1)[0]; keep1=rng.choice(idx1,size=max(35,len(idx0)//6),replace=False); keep=np.r_[idx0,keep1]; X=X.iloc[keep].reset_index(drop=True); y=y.iloc[keep].reset_index(drop=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
models={'logistic':Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=3000))]),'balanced_logistic':Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=3000,class_weight='balanced'))]),'balanced_rf':RandomForestClassifier(n_estimators=350,class_weight='balanced',random_state=42,n_jobs=-1)}
out={}; curves={}
for n,m in models.items():
 m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); out[n]={'average_precision':float(average_precision_score(yte,p)),'f1':float(f1_score(yte,pred))}; curves[n]=precision_recall_curve(yte,p)[:2]
Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
plt.figure(figsize=(7,5)); vals=y.value_counts().sort_index(); plt.bar([str(i) for i in vals.index],vals.values); plt.xlabel('Class'); plt.ylabel('Count'); plt.title('Controlled imbalance from real observations'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
plt.figure(figsize=(7,5));
for n,(prec,rec) in curves.items(): plt.plot(rec,prec,label=n)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-recall comparison'); plt.legend(); plt.tight_layout(); plt.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(); print(json.dumps(out,indent=2))