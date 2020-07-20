# Date:   09-Feb-2018
# Author: Satie
# 
# For running Decomposer in Python backend.

R_Decomposer <- function(d, tune_lam, lam_num, tune_gam, gam_num, K = 5, tol = 1e-5, max_iter = 300, eps = 1e-4)
{
  
    library("PythonInR")
    
    if (PythonInR::pyIsConnected())

    {

    PythonInR::pyExec("import Cov_Decomp as cd")
    PythonInR::pyExec("import numpy")

    # Pass data/params to Python and establish instance.

    PythonInR::pySet("_data", d, useNumpy = TRUE)

    class(tune_lam) <- "list"
    class(tune_gam) <- "list"

    PythonInR::pySet("_tune_lam", tune_lam) # list of floats
    PythonInR::pySet("_tune_gam", tune_gam) # list of floats
    PythonInR::pySet("_lam_num", lam_num)
    PythonInR::pySet("_gam_num", gam_num)

    PythonInR::pyExec("_tune_lam = tuple(_tune_lam + [int(_lam_num)])")
    PythonInR::pyExec("_tune_gam = tuple(_tune_gam + [int(_gam_num)])")

    class(max_iter) <- "integer"
    class(K)        <- "integer"

    PythonInR::pySet("_tol", tol) # float
    PythonInR::pySet("_max_iter", max_iter) # integer
    PythonInR::pySet("_eps", eps) # float
    PythonInR::pySet("_K", K) # integer

    PythonInR::pyExec("_K = int(_K)")

    PythonInR::pyExec("Decomposer = cd.Decomposer(_data)")

    # Main session

    PythonInR::pyExec("f, t = Decomposer.Estimator(_tune_lam, _tune_gam, K = _K, tol = _tol, max_iter = _max_iter, eps = _eps)")

    # Fetch results

    f <- pyGet("f")
    t <- pyGet("t")

    res <- list("F" = f, "T" = t)

    return(res)

    }

    else
  
    {

    stop("Python kernel not connected.")

    }
 
}