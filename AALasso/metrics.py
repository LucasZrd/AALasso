import numpy as np

def rNMSE(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the rNMSE given two multivariate time series

    Args:
        x (np.ndarray): first multivariate time series
        y (np.ndarray): second multivariate time series

    Returns:
        float: rNMSE
    """
    return np.sqrt(
        np.sum(np.linalg.norm(x - y, axis=0) ** 2)
        / np.sum(np.linalg.norm(y, axis=0) ** 2)
    )

def TP(preds, truth, eps = 10e-2):
    tp = 0
    for pred,t in zip(preds,truth):
        tp += np.sum((np.abs(pred)>eps).astype(int)*(np.abs(t)>eps).astype(int))
    return tp

def FP(preds,truth, eps = 10e-2): 
    fp = 0
    for pred,t in zip(preds,truth):
        fp += np.sum((np.abs(pred)>eps).astype(int)*(np.abs(t)<eps).astype(int))
    return fp

def TN(preds,truth, eps = 10e-2):
    tn = 0
    for pred,t in zip(preds,truth):
        tn += np.sum((np.abs(pred)<eps).astype(int)*(np.abs(t)<eps).astype(int))
    return tn

def FN(preds,truth, eps = 10e-2): 
    fn = 0
    for pred,t in zip(preds,truth):
        fn += np.sum((np.abs(pred)<eps).astype(int)*(np.abs(t)>eps).astype(int))
    return fn


def P(tp,fp):
    if tp+fp != 0:
        return tp/(tp+fp)
    return 0

def R(tp,fn):
    if tp+fn != 0:
        return tp/(tp+fn)
    return 0


def compute_metrics(preds, truth, eps):
    tp = TP(preds,truth, eps= eps)
    fp = FP(preds,truth, eps = eps)
    tn = TN(preds,truth, eps = eps)
    fn = FN(preds,truth, eps = eps)
    p = P(tp,fp)
    r = R(tp,fn)
    if p != 0 and r !=0:
        return p, r, 2/(1/p + 1/r)
    return p, r, 0