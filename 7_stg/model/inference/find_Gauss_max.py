import numpy as np
from copy import deepcopy
from scipy import optimize
import delfi.distribution as dd
from find_pyloric import params_are_bounded

def find_maximum(pdf, num_dim=1):
    # get starting point. We choose the point that has highest probability of all means.
    # in other words, the value that optimizes alpha / det(cov), where alpha is the component strength.
    try: # if it works, then the pdf is a MoG
        det_cov = []
        for k in range(len(pdf.a)):
            det_cov.append(np.linalg.det(pdf.xs[k].S))
        index = np.argmax(np.asarray(pdf.a) / np.asarray(det_cov))
        point = deepcopy(pdf.xs[index].m)
        result = optimize.minimize(eval_log_pdf, point, args=(pdf, False), method="BFGS")
        return result.x
    except: # if it doesn't work, then the pdf is a MAF
        point = np.zeros(num_dim)
        result = optimize.minimize(eval_log_pdf, point, args=(pdf, True), method="BFGS")
        return result.x


def eval_log_pdf(point, pdf, check_prior):
    if check_prior:
        prior = dd.Uniform(-np.sqrt(3) * np.ones(len(point)), np.sqrt(3) * np.ones(len(point)))
        if not np.all(params_are_bounded(point, prior, normalized=True)):
            return 1e20
        else:
            try:
                output = -pdf.eval([point])
            except:
                output = -pdf.eval(point)
            return output
    else:
        try:
            output = -pdf.eval([point])
        except:
            output = -pdf.eval(point)
        return output