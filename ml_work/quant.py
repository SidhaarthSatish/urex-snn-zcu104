import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, json
from sklearn.metrics import average_precision_score
tag,pfile=sys.argv[1],sys.argv[2]
res=json.load(open(f"res_{tag}.json")); R=res["R"]; th=res["th"]
d=np.load(pfile); P=torch.from_numpy(d["P"]); Y=d["y"]; Tt=d["t"]; N,T,_,_,_=d["P"].shape
blk=Tt//60_000_000; s=(Tt-blk*60_000_000)/1e6; role=np.where(blk%5==4,2,np.where(blk%5==3,1,0)); ok=(role==0)|(s>1)
va=np.nonzero((role==1)&ok)[0]; te=np.nonzero((role==2)&ok)[0]
sd=torch.load(f"w_{tag}.pt")
def qw(w):                                   # per-output-channel symmetric int8
    s=w.abs().flatten(1).max(1)[0].clamp(min=1e-8)/127; sh=[-1]+[1]*(w.dim()-1)
    return (w/s.view(sh)).round().clamp(-127,127)*s.view(sh)
def qa(x,sc): return (x/sc).round().clamp(0,255)*sc
W={k:(qw(v) if k.endswith("weight") else v) for k,v in sd.items()}
def fwd(x,W,sc=None,rec=None):
    x=torch.clamp(x,0,3)/3; x=x.flatten(1,2)
    def A(h,n):
        if rec is not None: rec[n]=max(rec.get(n,0),h.max().item())
        return qa(h,sc[n]) if sc else h
    h=A(F.relu(F.conv2d(x,W["c1.weight"],W["c1.bias"],padding=1)),"a1")
    if R==32: h=F.max_pool2d(h,2)
    h=F.max_pool2d(A(F.relu(F.conv2d(h,W["c2.weight"],W["c2.bias"],padding=1)),"a2"),2)
    h=A(F.relu(F.linear(h.flatten(1),W["fc1.weight"],W["fc1.bias"])),"a3")
    return F.linear(h,W["fc2.weight"],W["fc2.bias"])
def run(ix,W,sc=None,rec=None):
    out=[]
    with torch.no_grad():
        for i in range(0,len(ix),512): out.append(torch.softmax(fwd(P[ix[i:i+512]].float(),W,sc,rec),1)[:,1].numpy())
    return np.concatenate(out)
def metr(y,p):
    pr=p>=th; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=(~pr&(y==1)).sum(); a=tp/max(tp+fp,1); b=tp/max(tp+fn,1)
    return dict(precision=float(a),recall=float(b),f1=float(2*a*b/max(a+b,1e-9)),AP=float(average_precision_score(y,p)))
fl=run(te,sd); rec={}; run(va,W,None,rec); sc={k:v/255 for k,v in rec.items()}
q=run(te,W,sc)
out=dict(tag=tag,float32=metr(Y[te],fl),int8_ptq=metr(Y[te],q),agreement=float(((fl>=th)==(q>=th)).mean()))
print(json.dumps(out,indent=1)); json.dump(out,open(f"quant_{tag}.json","w"))
