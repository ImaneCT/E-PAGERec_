import argparse
import json
import math
import os
import random
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg, fh=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# data
def read_lgcn(path):
    """Reads 'u i1 i2 ...' lines."""
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            d[int(parts[0])] = [int(x) for x in parts[1:]]
    return d


class Data:
    def __init__(self, root, valid_ratio, seed, train_frac=1.0):
        train_all = read_lgcn(os.path.join(root, "train.txt"))
        test = read_lgcn(os.path.join(root, "test.txt"))

        self.n_users = max(max(train_all), max(test) if test else 0) + 1
        self.n_items = 1 + max(
            max(max(v) for v in train_all.values()),
            max(max(v) for v in test.values()) if test else 0,
        )

        # validation split (per user)
        rng = np.random.RandomState(seed)
        self.train, self.valid = {}, {}
        for u, items in train_all.items():
            items = np.unique(np.array(items, dtype=np.int64))
            n_val = int(round(len(items) * valid_ratio))
            # only users with at least 3 items get validation items
            if valid_ratio > 0 and len(items) >= 3 and n_val >= 1:
                perm = rng.permutation(len(items))
                self.valid[u] = items[perm[:n_val]]
                self.train[u] = items[perm[n_val:]]
            else:
                self.train[u] = items
        self.test = {u: np.unique(np.array(v, dtype=np.int64)) for u, v in test.items()}


        if train_frac < 1.0:
            rng2 = np.random.RandomState(seed + 7)
            for u, items in self.train.items():
                keep = max(1, int(round(len(items) * train_frac)))
                self.train[u] = items[rng2.permutation(len(items))[:keep]]

  
        rows = np.concatenate([np.full(len(v), u) for u, v in self.train.items()])
        cols = np.concatenate([v for v in self.train.values()])
        self.R = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        self.train_u = rows.astype(np.int64)
        self.train_i = cols.astype(np.int64)
        self.pos_keys = np.sort(self.train_u * self.n_items + self.train_i)


        vr = np.concatenate([np.full(len(v), u) for u, v in self.valid.items()])
        vc = np.concatenate([v for v in self.valid.values()])
        self.Rv = sp.csr_matrix(
            (np.ones(len(vr), dtype=np.float32), (vr, vc)),
            shape=(self.n_users, self.n_items),
        )
        self.n_train = len(rows)

    def summary(self):
        return (f"users={self.n_users} items={self.n_items} train={self.n_train} "
                f"valid={sum(len(v) for v in self.valid.values())} "
                f"test={sum(len(v) for v in self.test.values())} "
                f"density={self.n_train / (self.n_users * self.n_items):.5f}")

    def sample_negatives(self, users, n_retry=5):
        """Uniform negatives, resampled when they hit a training positive."""
        neg = np.random.randint(0, self.n_items, size=len(users))
        for _ in range(n_retry):
            keys = users * self.n_items + neg
            idx = np.searchsorted(self.pos_keys, keys)
            idx[idx >= len(self.pos_keys)] = len(self.pos_keys) - 1
            bad = self.pos_keys[idx] == keys
            if not bad.any():
                break
            neg[bad] = np.random.randint(0, self.n_items, size=bad.sum())
        return neg


def normalized_adjacency(R, n_users, n_items, device):
    A = sp.bmat([[None, R], [R.T, None]], format="coo").astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    d_inv = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
    A = sp.diags(d_inv) @ A @ sp.diags(d_inv)
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, (n_users + n_items,) * 2).coalesce().to(device)


# item-item neighborhood
def build_item_neighborhood(R, topk, chunk=1024):
    du = np.asarray(R.sum(1)).ravel()
    di = np.asarray(R.sum(0)).ravel()
    du_is = np.power(du, -0.5, where=du > 0, out=np.zeros_like(du))
    di_is = np.power(di, -0.5, where=di > 0, out=np.zeros_like(di))
    Rt = (sp.diags(du_is) @ R @ sp.diags(di_is)).astype(np.float32).tocsr()
    RtT = Rt.T.tocsr()
    n_items = R.shape[1]
    rows, cols, vals = [], [], []
    for s in range(0, n_items, chunk):
        e = min(s + chunk, n_items)
        blk = (RtT[s:e] @ Rt).toarray()
        blk[np.arange(e - s), np.arange(s, e)] = 0.0
        k = min(topk, n_items - 1)
        idx = np.argpartition(-blk, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(blk, idx, axis=1)
        keep = v > 0
        r = np.repeat(np.arange(s, e), k).reshape(e - s, k)
        rows.append(r[keep]); cols.append(idx[keep]); vals.append(v[keep])
    P = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
    )
    return P


# communities
@torch.no_grad()
def spherical_kmeans(X, K, iters, init=None, gen=None):
    X = F.normalize(X, dim=1)
    N = X.size(0)
    if init is not None and init.shape == (K, X.size(1)):
        C = init.clone()
    else:
        C = X[torch.randperm(N, generator=gen)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        counts = torch.bincount(a, minlength=K)
        Cn = torch.zeros_like(C).index_add_(0, a, X)
        empty = counts == 0
        if empty.any():  # re-seed empty clusters
            ridx = torch.randint(0, N, (int(empty.sum()),), generator=gen).to(X.device)
            Cn[empty] = X[ridx]
        C = F.normalize(Cn, dim=1)
    a = (X @ C.T).argmax(1)
    return a, C


def segment_mean(x, a, K):
    """Mean of rows of x per community (differentiable w.r.t. x)."""
    s = torch.zeros(K, x.size(1), device=x.device, dtype=x.dtype).index_add(0, a, x)
    cnt = torch.bincount(a, minlength=K).clamp(min=1).to(x.dtype).unsqueeze(1)
    return s / cnt


class Communities:
    def __init__(self, args, n_users, n_items, A, device):
        self.mode = args.comm
        self.Ku, self.Ki = args.k_user, args.k_item
        self.n_users, self.n_items = n_users, n_items
        self.A, self.p = A, args.pagec_p
        self.iters = args.kmeans_iters
        self.device = device
        self.gen = torch.Generator().manual_seed(args.seed)
        self.algo = args.cluster_algo
        self.knn = args.pagec_knn
        # E-PAGEC state
        self.params = []
        self.tau_g = args.tau_g
        self.Zu = self.Zi = None            
        self.WXu = self.WXi = None          
        self._cache = None
        if self.algo == "epagec":
            self.Qu = torch.nn.Parameter(torch.eye(args.dim, device=device))
            self.Qi = torch.nn.Parameter(torch.eye(args.dim, device=device))
            self.params = [self.Qu, self.Qi]
        self.au = self.ai = None          
        self.proto_u = self.proto_i = None  # prototypes for proto_loss
        self.cu = self.ci = None          # k-means centroids (warm start)
        self.history = []

    @property
    def active(self):
        return self.mode != "none" and self.au is not None

    @torch.no_grad()
    def diffused_features(self, E):
        if self.p == 0:
            # p = 0: k-means on the embeddings, no diffusion
            return F.normalize(E, dim=1)

        def W(x):
            return torch.sparse.mm(self.A, torch.sparse.mm(self.A, x))
        h = W(E)
        out = F.normalize(h, dim=1)
        for _ in range(self.p - 1):
            h = W(h)
        if self.p > 1:
            out = out + F.normalize(h, dim=1)
        return out

    @torch.no_grad()
    def knn_graph(self, X, k, chunk=2048):
        Xn = F.normalize(X, dim=1)
        n = Xn.size(0)
        rows, cols, vals = [], [], []
        for s0 in range(0, n, chunk):
            sim = Xn[s0: s0 + chunk] @ Xn.T
            r = torch.arange(s0, min(s0 + chunk, n), device=X.device)
            sim[torch.arange(len(r), device=X.device), r] = -1.0     
            v, c = sim.topk(k, dim=1)
            rows.append(r.unsqueeze(1).expand(-1, k).reshape(-1))
            cols.append(c.reshape(-1)); vals.append(v.clamp(min=0).reshape(-1))
        rows, cols, vals = torch.cat(rows), torch.cat(cols), torch.cat(vals)
        # symmetrize
        idx = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], 1)
        W = torch.sparse_coo_tensor(idx, torch.cat([vals, vals]) / 2, (n, n)).coalesce()
        deg = torch.sparse.sum(W, 1).to_dense().clamp(min=1e-8)
        i = W.indices()
        return torch.sparse_coo_tensor(i, W.values() / deg[i[0]], (n, n)).coalesce()


    def ep_features(self, E):
        nu = self.n_users
        M = E
        for _ in range(max(self.p, 1)):
            M = torch.sparse.mm(self.A, torch.sparse.mm(self.A, M))   
        M = F.normalize(M, dim=1)
        Bu, Bi = M[:nu] @ self.Qu, M[nu:] @ self.Qi
        SB = torch.sparse.mm(self.A, torch.sparse.mm(self.A, torch.cat([Bu, Bi], 0)))
        SBu = SB[:nu] + torch.sparse.mm(self.WXu, Bu)
        SBi = SB[nu:] + torch.sparse.mm(self.WXi, Bi)
        return F.normalize(SBu, dim=1), F.normalize(SBi, dim=1), M, Bu, Bi

    def st_assign(self, H, Z):
        L = H @ Z.T / self.tau_g
        Gs = torch.softmax(L, dim=1)
        idx = L.argmax(1)
        Gh = torch.zeros_like(Gs).scatter_(1, idx.unsqueeze(1), 1.0)
        return Gh + Gs - Gs.detach(), idx, Gs

    @torch.no_grad()
    def proto_step(self, H, a, K):
        Zs = torch.zeros(K, H.size(1), device=H.device).index_add_(0, a, H)
        cnt = torch.bincount(a, minlength=K)
        empty = cnt == 0
        if empty.any():  
            ridx = torch.randint(0, H.size(0), (int(empty.sum()),), generator=self.gen).to(H.device)
            Zs[empty] = H[ridx]
        return F.normalize(Zs, dim=1)

    @torch.no_grad()
    def epagec_refresh(self, E):
        nu = self.n_users
        Ed = E.detach()
        self.WXu = self.knn_graph(Ed[:nu], self.knn)
        self.WXi = self.knn_graph(Ed[nu:], self.knn)
        Hu, Hi, _, _, _ = self.ep_features(Ed)
        if self.Zu is None:   # first refresh: k-means
            au, self.Zu = spherical_kmeans(Hu, self.Ku, self.iters, None, self.gen)
            ai, self.Zi = spherical_kmeans(Hi, self.Ki, self.iters, None, self.gen)
        else:
            for _ in range(2):
                au = (Hu @ self.Zu.T).argmax(1)
                ai = (Hi @ self.Zi.T).argmax(1)
                self.Zu = self.proto_step(Hu, au, self.Ku)
                self.Zi = self.proto_step(Hi, ai, self.Ki)
            au = (Hu @ self.Zu.T).argmax(1)
            ai = (Hi @ self.Zi.T).argmax(1)
        return au, ai

    def epagec_convolve(self, E, beta):
        nu = self.n_users
        Hu, Hi, M, Bu, Bi = self.ep_features(E)
        Gu, au, Gsu = self.st_assign(Hu, self.Zu)
        Gi, ai, Gsi = self.st_assign(Hi, self.Zi)
        Eu, Ei = E[:nu], E[nu:]
        if beta > 0:
            Cu = (Gu.T @ Eu) / Gu.sum(0).unsqueeze(1).clamp(min=1e-8)
            Ci = (Gi.T @ Ei) / Gi.sum(0).unsqueeze(1).clamp(min=1e-8)
            Eu = Eu + beta * (Gu @ Cu)
            Ei = Ei + beta * (Gi @ Ci)
        self._cache = {"Hu": Hu, "Hi": Hi, "au": au, "ai": ai, "Gsu": Gsu, "Gsi": Gsi,
                       "M": M, "Bu": Bu, "Bi": Bi}
        return torch.cat([Eu, Ei], 0)

    def ep_loss(self, users, items):
        c, nu = self._cache, self.n_users
        hu, hi = c["Hu"][users], c["Hi"][items]
        compact = ((hu - self.Zu[c["au"][users]]) ** 2).sum(1).mean() + \
                  ((hi - self.Zi[c["ai"][items]]) ** 2).sum(1).mean()
        Mu, Mi = c["M"][:nu][users], c["M"][nu:][items]
        recon = ((Mu - c["Bu"][users] @ self.Qu.T) ** 2).sum(1).mean() + \
                ((Mi - c["Bi"][items] @ self.Qi.T) ** 2).sum(1).mean()
        pu, pi = c["Gsu"][users].mean(0), c["Gsi"][items].mean(0)
        bal = (pu * torch.log(pu * self.Ku + 1e-12)).sum() + (pi * torch.log(pi * self.Ki + 1e-12)).sum()
        return compact + recon, bal

    @torch.no_grad()
    def refresh(self, E, epoch):
        if self.mode == "none":
            return
        if self.mode == "static" and self.au is not None:
            return  # static: computed once
        if self.algo == "epagec":
            au, ai = self.epagec_refresh(E)
        else:
            M = self.diffused_features(E)
            Mu, Mi = M[: self.n_users], M[self.n_users:]
            au, self.cu = spherical_kmeans(Mu, self.Ku, self.iters, self.cu, self.gen)
            ai, self.ci = spherical_kmeans(Mi, self.Ki, self.iters, self.ci, self.gen)
        if self.mode == "random":
            # random membership, same community sizes
            au = au[torch.randperm(self.n_users, generator=self.gen).to(au.device)]
            ai = ai[torch.randperm(self.n_items, generator=self.gen).to(ai.device)]

        # share of users and items that changed community since the last refresh
        stats = {"epoch": epoch}
        if self.au is not None:
            stats["moved_u"] = float((self.au.cpu().numpy() != au.cpu().numpy()).mean())
            stats["moved_i"] = float((self.ai.cpu().numpy() != ai.cpu().numpy()).mean())
        self.history.append(stats)

        self.au, self.ai = au, ai
        # prototypes for proto_loss
        En = F.normalize(E, dim=1)
        self.proto_u = F.normalize(segment_mean(En[: self.n_users], au, self.Ku), dim=1)
        self.proto_i = F.normalize(segment_mean(En[self.n_users:], ai, self.Ki), dim=1)
        return stats

    def convolve(self, E, beta):
        if self.algo == "epagec":
            return self.epagec_convolve(E, beta)
        Eu, Ei = E[: self.n_users], E[self.n_users:]
        Eu = Eu + beta * segment_mean(Eu, self.au, self.Ku)[self.au]
        Ei = Ei + beta * segment_mean(Ei, self.ai, self.Ki)[self.ai]
        return torch.cat([Eu, Ei], 0)

    def proto_loss(self, E, users, items, tau):
        eu = F.normalize(E[users], dim=1)
        ei = F.normalize(E[self.n_users + items], dim=1)
        lu = F.cross_entropy(eu @ self.proto_u.T / tau, self.au[users])
        li = F.cross_entropy(ei @ self.proto_i.T / tau, self.ai[items])
        return lu + li


# encoder
class Encoder(torch.nn.Module):
    """XSimGCL encoder."""

    def __init__(self, n_nodes, args, A):
        super().__init__()
        self.emb = torch.nn.Parameter(torch.empty(n_nodes, args.dim))
        torch.nn.init.xavier_uniform_(self.emb)
        self.A = A
        self.L = args.layers
        self.eps = args.eps

    def forward(self, perturb=False, comm=None, beta=0.0):
        e = self.emb
        layers = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e)
            if perturb:
                noise = F.normalize(torch.rand_like(e), dim=1)
                e = e + torch.sign(e) * noise * self.eps
            layers.append(e)
        out = torch.stack(layers[1:], 0).mean(0)
        if comm is not None and comm.active and (beta > 0 or comm.algo == "epagec"):
            out = comm.convolve(out, beta)
        return out, layers[1]   # first layer: view for the contrastive loss


def info_nce(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.T / tau
    return F.cross_entropy(logits, torch.arange(a.size(0), device=a.device))


# evaluation
def zrow(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)


def combo_key(g, c):
    return f"{g:g}|{c:g}"


def build_combos(gammas, gcs):
    out = []
    for g in gammas:
        for c in gcs:
            if (g < 0 and c != 0) or (c < 0 and g != 0):
                continue
            out.append((g, c))
    return out


@torch.no_grad()
def community_scores(comm, data, device):
    """Share of the members of each user community who interacted with each item."""
    au = comm.au.cpu().numpy()
    C = sp.csr_matrix((np.ones(len(au), dtype=np.float32), (au, np.arange(len(au)))),
                      shape=(comm.Ku, len(au)))
    M = (C @ data.R).toarray()
    M /= np.maximum(np.bincount(au, minlength=comm.Ku), 1)[:, None]
    return torch.from_numpy(M.astype(np.float32)).to(device)


@torch.no_grad()
def evaluate(model, comm, data, args, target, mask_mats, P_t, combos, device, k=20):
    model.eval()
    E, _ = model(perturb=False, comm=comm, beta=args.beta)
    U, I = E[: data.n_users], E[data.n_users:]
    users = np.array(sorted(target.keys()))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    keys = [combo_key(g, c) for g, c in combos]
    res = {key: {"recall@20": 0.0, "ndcg@20": 0.0} for key in keys}
    per_user = {key: np.zeros(len(users)) for key in keys}

    need_nb = P_t is not None and any(g != 0 for g, _ in combos)
    need_com = comm.active and any(c != 0 for _, c in combos)
    Mc = community_scores(comm, data, device) if need_com else None

    for s in range(0, len(users), args.eval_batch):
        ub = users[s: s + args.eval_batch]
        ub_t = torch.from_numpy(ub).to(device)
        base = U[ub_t] @ I.T
        zb = zrow(base)
        if need_nb:
            Rb = torch.from_numpy(data.R[ub].toarray()).to(device)
            zg = zrow(torch.sparse.mm(P_t, Rb.T).T)
        if need_com:
            zc = zrow(Mc[comm.au[ub_t]])
        mrow, mcol = [], []
        for M in mask_mats:
            sub = M[ub].tocoo()
            mrow.append(sub.row); mcol.append(sub.col)
        mrow = torch.from_numpy(np.concatenate(mrow).astype(np.int64)).to(device)
        mcol = torch.from_numpy(np.concatenate(mcol).astype(np.int64)).to(device)
        tgt = torch.zeros(len(ub), data.n_items, dtype=torch.bool, device=device)
        tr = np.concatenate([np.full(len(target[u]), j) for j, u in enumerate(ub)])
        tc = np.concatenate([target[u] for u in ub])
        tgt[torch.from_numpy(tr.astype(np.int64)).to(device),
            torch.from_numpy(tc.astype(np.int64)).to(device)] = True
        n_t = np.array([len(target[u]) for u in ub])
        idcg = np.array([disc[: min(n, k)].sum() for n in n_t])

        for (g, c), key in zip(combos, keys):
            use_g = need_nb and g != 0
            use_c = need_com and c != 0
            if g < 0 and use_g:
                sc = zg.clone()                      # neighborhood score alone
            elif c < 0 and use_c:
                sc = zc.clone()                      # community score alone
            elif not use_g and not use_c:
                sc = base.clone()                    # embedding score alone
            else:
                sc = zb.clone()
                if use_g:
                    sc += g * zg
                if use_c:
                    sc += c * zc
            sc[mrow, mcol] = -float("inf")
            top = sc.topk(k, dim=1).indices
            h = tgt.gather(1, top).float().cpu().numpy()
            nd = (h * disc[:k]).sum(1) / idcg
            res[key]["recall@20"] += (h.sum(1) / n_t).sum()
            res[key]["ndcg@20"] += nd.sum()
            per_user[key][s: s + len(ub)] = nd
    for key in keys:
        for m in res[key]:
            res[key][m] /= len(users)
    model.train()
    return res, per_user, users


def compact_line(res, combos):
    """Best NDCG@20 of each variant, for the log."""
    def best_of(cond):
        ks = [combo_key(g, c) for g, c in combos if cond(g, c)]
        if not ks:
            return "  -   "
        k = max(ks, key=lambda k: res[k]["ndcg@20"])
        return f"{res[k]['ndcg@20']:.4f} [{k}]"
    return (f"full={best_of(lambda g, c: g >= 0 and c >= 0)} | "
            f"w/o community score={best_of(lambda g, c: g >= 0 and c == 0)} | "
            f"community score alone={best_of(lambda g, c: c < 0)}   (NDCG@20 [gn|gc])")


@torch.no_grad()
def save_snapshot(out, tag, E, comm, data):
    """Saves item embeddings and item communities."""
    Ei = E[data.n_users:].detach().float().cpu().numpy().astype(np.float16)
    np.savez_compressed(os.path.join(out, f"emb_{tag}.npz"), emb=Ei,
                        comm=comm.ai.cpu().numpy().astype(np.int32),
                        item_deg=np.asarray(data.R.sum(0)).ravel().astype(np.int32))


# training
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data", required=True, help="folder with train.txt / test.txt")
    ap.add_argument("--out", default="runs/tmp")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="keep this fraction of each user's train items")
    ap.add_argument("--seed", type=int, default=2026)
    # encoder
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lambda_cl", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--tau_cl", type=float, default=0.15)
    # communities
    ap.add_argument("--comm", choices=["none", "dynamic", "static", "random"], default="dynamic")
    ap.add_argument("--cluster_algo", choices=["epagec", "kmeans"], default="epagec",
                    help="epagec: E-PAGEC; kmeans: spherical k-means on the diffused embeddings")
    ap.add_argument("--k_user", type=int, default=200)
    ap.add_argument("--k_item", type=int, default=200)
    ap.add_argument("--pagec_p", type=int, default=2, help="propagation power p (0 = no diffusion)")
    ap.add_argument("--pagec_knn", type=int, default=20)
    ap.add_argument("--tau_g", type=float, default=0.1, help="assignment temperature")
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--comm_start", type=int, default=3, help="epoch of the first refresh")
    ap.add_argument("--refresh_every", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1, help="community convolution weight")
    ap.add_argument("--lambda_proto", type=float, default=1e-2, help="clustering loss weight")
    ap.add_argument("--lambda_bal", type=float, default=1e-2, help="balance loss weight")
    ap.add_argument("--tau_proto", type=float, default=0.1, help="k-means variants only")
    # ranking
    ap.add_argument("--gammas", default="0,0.1,0.15,0.2,0.3,0.4,-1",
                    help="neighborhood score weights (-1 = that score alone)")
    ap.add_argument("--gammas_c", default="0,0.01,0.02,0.05,0.1,0.2,-1",
                    help="community score weights (-1 = that score alone)")
    ap.add_argument("--lin_topk", type=int, default=256)
    ap.add_argument("--save_emb_epochs", default="",
                    help="save item embeddings and communities at these refresh epochs, e.g. 3,7,15,final")
    # evaluation
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--eval_batch", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=6, help="evaluations without improvement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.valid_ratio <= 0:
        ap.error("--valid_ratio must be > 0")

    os.makedirs(args.out, exist_ok=True)
    fh = open(os.path.join(args.out, "log.txt"), "w")
    set_seed(args.seed)
    device = torch.device(args.device)
    gammas = [float(g) for g in args.gammas.split(",")]
    save_epochs = set(x.strip() for x in args.save_emb_epochs.split(",") if x.strip())
    gcs = [float(c) for c in args.gammas_c.split(",")] if args.comm != "none" else [0.0]
    combos = build_combos(gammas, gcs)
    keys = [combo_key(g, c) for g, c in combos]

    # data and graph
    data = Data(args.data, args.valid_ratio, args.seed, args.train_frac)
    log("data: " + data.summary(), fh)
    A = normalized_adjacency(data.R, data.n_users, data.n_items, device)
    P_t = None
    if any(g != 0 for g in gammas):
        t = time.time()
        P = build_item_neighborhood(data.R, args.lin_topk)
        PT = P.T.tocoo()
        P_t = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([PT.row, PT.col]).astype(np.int64)),
            torch.from_numpy(PT.data.astype(np.float32)), PT.shape).coalesce().to(device)
        log(f"item neighborhood: nnz={P.nnz} ({time.time() - t:.1f}s)", fh)

    model = Encoder(data.n_users + data.n_items, args, A).to(device)
    comm = Communities(args, data.n_users, data.n_items, A, device)
    opt = torch.optim.Adam(list(model.parameters()) + comm.params, lr=args.lr)

    test_masks = [data.R, data.Rv]
    best = {k: {"score": -1, "epoch": -1, "test": None, "per_user": None} for k in keys}
    curve = []
    bad_evals = 0
    n_batches = int(math.ceil(data.n_train / args.batch))

    for epoch in range(1, args.epochs + 1):
        # community refresh
        if args.comm != "none" and epoch >= args.comm_start and \
                (epoch - args.comm_start) % args.refresh_every == 0:
            with torch.no_grad():
                model.eval()
                E, _ = model(perturb=False, comm=comm, beta=args.beta)
                model.train()
            st = comm.refresh(E, epoch)
            if st:
                log("communities: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in st.items()}), fh)
            if str(epoch) in save_epochs and comm.ai is not None:
                save_snapshot(args.out, f"ep{epoch}", E, comm, data)

        t0 = time.time()
        perm = np.random.permutation(data.n_train)
        tot = {"bpr": 0.0, "cl": 0.0, "clu": 0.0}
        for b in range(n_batches):
            idx = perm[b * args.batch: (b + 1) * args.batch]
            u_np, i_np = data.train_u[idx], data.train_i[idx]
            j_np = data.sample_negatives(u_np)
            u = torch.from_numpy(u_np).to(device)
            i = torch.from_numpy(i_np).to(device)
            j = torch.from_numpy(j_np).to(device)

            # propagation + community convolution
            E, E_cl = model(perturb=True, comm=comm, beta=args.beta)
            eu, ei, ej = E[u], E[data.n_users + i], E[data.n_users + j]
            rec = F.softplus(-((eu * ei).sum(1) - (eu * ej).sum(1))).mean()

            ego = model.emb
            reg = args.reg * (ego[u].norm(2) + ego[data.n_users + i].norm(2) +
                              ego[data.n_users + j].norm(2)) / len(idx)
            loss = rec + reg

            # contrastive loss
            uu, ui = torch.unique(u), torch.unique(i)
            cl = torch.zeros((), device=device)
            if args.lambda_cl > 0:
                cl = info_nce(E[uu], E_cl[uu], args.tau_cl) + \
                    info_nce(E[data.n_users + ui], E_cl[data.n_users + ui], args.tau_cl)
            loss = loss + args.lambda_cl * cl

            # community losses
            pr = torch.zeros((), device=device)
            if comm.active and comm.algo == "epagec":
                pr, bal = comm.ep_loss(uu, ui)
                loss = loss + args.lambda_proto * pr + args.lambda_bal * bal
            elif comm.active and args.lambda_proto > 0:
                pr = comm.proto_loss(E, uu, ui, args.tau_proto)
                loss = loss + args.lambda_proto * pr

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["bpr"] += rec.item(); tot["cl"] += cl.item(); tot["clu"] += pr.item()

        msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in tot.items())
        log(f"epoch {epoch:3d} | {msg} | {time.time() - t0:.1f}s", fh)

      
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            res, _, _ = evaluate(model, comm, data, args, data.valid, [data.R], P_t, combos, device)
            log("  [valid] " + compact_line(res, combos), fh)
            curve.append({"epoch": epoch, "valid": res})

            improved = [k for k in keys if res[k]["ndcg@20"] > best[k]["score"]]
            if improved:
                bad_evals = 0
                tres, tpu, test_users = evaluate(model, comm, data, args, data.test,
                                                 test_masks, P_t, combos, device)
                for k in improved:
                    best[k] = {"score": res[k]["ndcg@20"], "epoch": epoch,
                               "test": tres[k], "per_user": tpu[k]}
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    log("early stop", fh)
                    break

    if "final" in save_epochs and comm.ai is not None:
        with torch.no_grad():
            model.eval()
            E, _ = model(perturb=False, comm=comm, beta=args.beta)
        save_snapshot(args.out, "final", E, comm, data)

    def pick(cond):
        cand = [(g, c) for g, c in combos if cond(g, c)]
        if not cand:
            return None
        return combo_key(*max(cand, key=lambda gc: best[combo_key(*gc)]["score"]))

    rows = {
        "full": pick(lambda g, c: g >= 0 and c >= 0),
        "no_comm_channel": pick(lambda g, c: g >= 0 and c == 0),   # w/o community score
        "comm_only": pick(lambda g, c: c < 0),                      # community score alone
    }
    summary = {
        "args": vars(args),
        "rows": {name: {"combo": k, "valid": best[k]["score"], "epoch": best[k]["epoch"],
                        "test": best[k]["test"]} for name, k in rows.items() if k},
        "community_history": comm.history,
        "curve": curve,
    }
    summary["final_test"] = best[rows["full"]]["test"]

    arrays = {"users": test_users}
    for name in ("full", "no_comm_channel"):
        if rows[name]:
            arrays["ndcg20_" + name] = best[rows[name]]["per_user"]
    np.savez(os.path.join(args.out, "per_user.npz"), **arrays)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    labels = {"full": "E-PAGERec", "no_comm_channel": "w/o community score",
              "comm_only": "community score alone"}
    log("test (selection on validation)", fh)
    for name, r in summary["rows"].items():
        t = r["test"]
        log(f"  {labels[name]:<24} [{r['combo']:>9}]  Recall@20={t['recall@20']:.4f} "
            f"NDCG@20={t['ndcg@20']:.4f}", fh)
    fh.close()


if __name__ == "__main__":
    main()
