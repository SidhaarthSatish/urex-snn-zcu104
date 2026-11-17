import numpy as np, sys, time
import paths; D=paths.MASKED
def load():
    d=np.load(D); return d["x"].astype(np.int32),d["y"].astype(np.int32),d["p"].astype(np.int8),d["t"].astype(np.int64),d["lab"]
def sample_idx(t,lab,n,cap_per_s,seed=0):
    rng=np.random.default_rng(seed); sec=t//1_000_000
    perm=rng.permutation(len(t)); cnt={}
    out=[]
    # per-second cap, keep first cap by random order
    s=sec[perm]; order=np.argsort(s,kind="stable"); ps=perm[order]; ss=s[order]
    start=np.r_[0,np.nonzero(np.diff(ss))[0]+1,len(ss)]
    for a,b in zip(start[:-1],start[1:]): out.append(ps[a:min(b,a+cap_per_s)])
    out=np.concatenate(out); rng.shuffle(out); return np.sort(out[:n])
def make(x,y,p,t,idx,R,W_us,T):
    N=len(idx); P=np.zeros((N,T,2,R,R),np.uint8); h=R//2
    lo_all=np.searchsorted(t,t[idx]-W_us)
    for k,i in enumerate(idx):
        lo=lo_all[k]; sl=slice(lo,i+1)
        xs=x[sl]-x[i]+h; ys=y[sl]-y[i]+h
        m=(xs>=0)&(xs<R)&(ys>=0)&(ys<R)
        if not m.any(): continue
        b=((t[sl][m]-(t[i]-W_us))*T//(W_us+1)).clip(0,T-1)
        np.add.at(P[k],(b,p[sl][m],ys[m],xs[m]),1)
    return P
if __name__=="__main__":
    R,Wms,T,n,cap,out=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6]
    x,y,p,t,lab=load(); idx=sample_idx(t,lab,n,cap)
    t0=time.time(); P=make(x,y,p,t,idx,R,int(Wms*1000),T); print("made",P.shape,time.time()-t0)
    np.savez_compressed(out,P=P,y=(lab[idx]>0).astype(np.uint8),trk=lab[idx],t=t[idx],xy=np.c_[x[idx],y[idx]])
    print("pos frac",(lab[idx]>0).mean())
