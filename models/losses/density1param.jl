# Custom PySR loss for models/density.py — 1-parameter density model.
#
# For each candidate density shape h, find optimal (a, b, c, d) per galaxy:
#   Step 1 – outer loop: sweep d on a log grid.
#   Step 2 – given d, evaluate h(r/d), integrate to the DM column
#            dm_col_i = (1/r_i) ∫₀^{r_i} h(r'/d) r'² dr'  (cumulative trapezoid
#            with a virtual point at r=0), then solve IRLS for a, b, c.
#   Step 3 – fine local refinement (20 evals) around the best grid d, so exact
#            constant-free forms are not penalised vs BFGS-tunable ones.
#   Step 4 – return total Student-t NLL at the best d per galaxy.
#
# No origin penalty: V²_DM(0)=0 automatically via the integral.
# Data MUST be sorted by (galaxy_code, r) in Python before fitting.
#
# Hyperparameters appear as double-underscored UPPERCASE placeholder tokens,
# filled in by models/julia_losses.py.
function physicsloss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    n_pts      = length(dataset.y)
    n_gal      = Int(maximum(dataset.X[2, :]))
    total_loss = L(0)

    # Student-t degrees of freedom (matching mcmc_fit.py: ν=3)
    ν  = L(__NU_T__)
    ν1 = ν + L(1)   # ν + 1, used in IRLS weight: w_irls = w * (ν+1) / (ν + w*r²)

    # Log-normal priors on upsilon_disk (a) and upsilon_bulge (b),
    # matching mcmc_fit.py: lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex).
    upsilon_wt = L(__UPSILON_WEIGHT__)
    σ_ml       = L(0.1 * log(10))   # 0.1 dex in natural log units
    μ_disk     = log(L(0.5))
    μ_bulge    = log(L(0.7))

    # ---- physics penalties (evaluated once; h(x) independent of d) ----
    # Non-negativity: integration clamps h<0 silently, so penalise Σmax(−h,0).
    # Inner log-slope: s = −d(ln h)/d(ln x) at x∈[0.001,0.01]; enforce 0 ≤ s ≤ 2.
    #   s<0 means density rises toward centre; s>2 means inner cusp steeper than r^-2.
    X_phys = zeros(T, 5, 7)
    X_phys[1, :] .= T[0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
    h_phys, ok_phys = eval_tree_array(tree.trees.f, X_phys, options)
    nonneg_pen = (ok_phys && all(isfinite, h_phys)) ? sum(max.(-h_phys, L(0))) : L(0)
    slope_pen  = if ok_phys && all(isfinite, h_phys) && h_phys[1] > T(1e-8) && h_phys[2] > T(1e-8)
        s_inner = log(h_phys[1] / h_phys[2]) / log(X_phys[1, 2] / X_phys[1, 1])
        max(-s_inner, T(0)) + max(s_inner - T(2), T(0))
    else
        L(0)
    end

    # ---- pre-compute per-galaxy index ranges (data sorted by galaxy_code) ----
    # O(n_pts) scan replaces O(n_gal × n_pts) BitVector masks per expression eval.
    g_start = zeros(Int, n_gal)
    g_end   = zeros(Int, n_gal)
    for i in 1:n_pts
        g = Int(dataset.X[2, i])
        if g_start[g] == 0
            g_start[g] = i
        end
        g_end[g] = i
    end

    for g in 1:n_gal
        g_start[g] == 0 && continue   # galaxy absent from dataset
        n_g = g_end[g] - g_start[g] + 1
        n_g < 3 && continue   # need at least 3 points for 3-parameter solve

        rng_g    = g_start[g]:g_end[g]
        r_g      = view(dataset.X, 1, rng_g)
        Vgas2_g  = view(dataset.X, 3, rng_g)   # sign(Vgas)·Vgas² (signed)
        Vdisk2_g = view(dataset.X, 4, rng_g)
        Vbul2_g  = view(dataset.X, 5, rng_g)
        y_g      = view(dataset.y, rng_g)
        w_g      = view(dataset.weights, rng_g)  # 1/(2·Vobs·errV)²
        resid_g  = y_g .- Vgas2_g   # target: a·Vd² + b·Vb² + c·dm_col

        # ---- pre-allocate work arrays ONCE per galaxy (outside d-loop) ----
        X_eval = Matrix{T}(undef, 5, n_g)
        X_eval[2, :] .= T(g)
        X_eval[3, :]  = Vgas2_g
        X_eval[4, :]  = Vdisk2_g
        X_eval[5, :]  = Vbul2_g

        A           = Matrix{T}(undef, n_g, 3)   # design matrix [Vd², Vb², dm_col]
        A_w         = Matrix{T}(undef, n_g, 3)   # sqrt(w)-weighted (initial WLS)
        A_irls      = Matrix{T}(undef, n_g, 3)   # IRLS-weighted design matrix
        r_work      = Vector{T}(undef, n_g)       # residual work vector
        w_irls      = Vector{T}(undef, n_g)       # Student-t IRLS weights
        W_irls_sqrt = Vector{T}(undef, n_g)       # sqrt(w_irls)
        resid_irls  = Vector{T}(undef, n_g)       # IRLS-weighted residual
        dm_col      = Vector{T}(undef, n_g)       # cumulative-trapezoid DM column

        A[:, 1]   = Vdisk2_g
        A[:, 2]   = Vbul2_g
        W_sqrt    = sqrt.(w_g)
        resid_w   = W_sqrt .* resid_g
        A_w[:, 1] = W_sqrt .* Vdisk2_g
        A_w[:, 2] = W_sqrt .* Vbul2_g

        # ---- nested helper: evaluate loss at a given log(d) ----
        # Returns L(Inf) on failure.  Captures all per-galaxy arrays.
        # Note: A[:, 3], A_w[:, 3], dm_col are mutated on each call.
        function eval_candidate_g(log_d_val, n_irls_local::Int=Int(__N_IRLS_FINE__))
            d_val = exp(log_d_val)
            X_eval[1, :] .= r_g ./ d_val

            # Evaluate density shape h(r/d)
            f_v, fl = eval_tree_array(tree.trees.f, X_eval, options)
            !fl && return L(Inf)
            any(!isfinite, f_v) && return L(Inf)

            # Cumulative trapezoid with virtual origin (r=0, h·r²=0)
            cum_I_loc = T(0); prev_fr2 = T(0); prev_r_loc = T(0)
            for i in 1:n_g
                r_i    = r_g[i]
                f_r2_i = f_v[i] * r_i * r_i
                cum_I_loc  += (f_r2_i + prev_fr2) * T(0.5) * (r_i - prev_r_loc)
                dm_col[i]   = cum_I_loc / r_i
                prev_fr2    = f_r2_i
                prev_r_loc  = r_i
            end
            any(!isfinite, dm_col) && return L(Inf)

            A[:, 3]   .= dm_col
            A_w[:, 3] .= W_sqrt .* dm_col
            maximum(abs, A) < T(1e-10) && return L(Inf)

            par = try
                A_w \ resid_w
            catch
                return L(Inf)
            end
            par = max.(par, L(0))

            for _ in 1:n_irls_local
                r_work .= A * par
                r_work .-= resid_g
                w_irls .= w_g .* ν1 ./ (ν .+ w_g .* r_work .^ 2)
                W_irls_sqrt .= sqrt.(w_irls)
                A_irls[:, 1] .= W_irls_sqrt .* Vdisk2_g
                A_irls[:, 2] .= W_irls_sqrt .* Vbul2_g
                A_irls[:, 3] .= W_irls_sqrt .* dm_col
                resid_irls .= W_irls_sqrt .* resid_g
                new_par = try
                    A_irls \ resid_irls
                catch
                    break
                end
                par = max.(new_par, L(0))
            end

            r_work .= A * par
            r_work .-= resid_g
            st = sum(log.(L(1) .+ w_g .* r_work .^ 2 ./ ν)) / n_g
            (isinf(st) || isnan(st)) && return L(Inf)

            pr = L(0)
            if upsilon_wt > L(0)
                a_opt = par[1]; b_opt = par[2]
                if a_opt > L(0)
                    pr += (log(a_opt) - μ_disk)^2 / (2 * σ_ml^2)
                else
                    pr += L(1000) * a_opt^2
                end
                has_bulge = maximum(abs, Vbul2_g) > T(1e-6)
                if has_bulge && b_opt > L(0)
                    pr += (log(b_opt) - μ_bulge)^2 / (2 * σ_ml^2)
                elseif has_bulge
                    pr += L(1000) * b_opt^2
                end
            end
            return st + upsilon_wt * pr
        end   # eval_candidate_g

        # ---- per-galaxy adaptive d range ----
        r_max_g    = maximum(r_g)
        d_min_g    = r_max_g / T(20)
        d_max_g    = max(r_max_g * T(10), T(50))
        log_d_lo_g = log(d_min_g)
        log_d_hi_g = log(d_max_g)

        # ---- Coarse grid search: log-spaced d (few IRLS iterations for speed) ----
        best_g     = L(Inf)
        best_log_d = (log_d_lo_g + log_d_hi_g) / T(2)
        for log_d in range(log_d_lo_g, log_d_hi_g; length=__N_D_GRID__)
            c = eval_candidate_g(log_d, __N_IRLS_COARSE__)
            if c < best_g
                best_g     = c
                best_log_d = log_d
            end
        end

        # ---- Fine local refinement around best_log_d ----
        # Scan 20 points within one coarse grid step of best_log_d.
        # This removes the BFGS-constant advantage: forms like c/(x²+a)
        # are exactly pISO at rescaled d — with near-continuous d-opt,
        # the exact constant-free form 1/(1+x²) ties them on loss and
        # PySR's complexity criterion correctly picks the simpler one.
        Δlog   = (log_d_hi_g - log_d_lo_g) / (__N_D_GRID__ - 1)
        lo_ref = max(log_d_lo_g, best_log_d - Δlog)
        hi_ref = min(log_d_hi_g, best_log_d + Δlog)
        for log_d in range(lo_ref, hi_ref; length=20)
            c = eval_candidate_g(log_d)
            if c < best_g; best_g = c; end
        end

        isinf(best_g) && (best_g = L(1e12))
        best_g += L(__WEIGHT_NONNEG__) * nonneg_pen + L(__WEIGHT_SLOPE__) * slope_pen
        total_loss += best_g
    end

    return total_loss / n_gal
end
