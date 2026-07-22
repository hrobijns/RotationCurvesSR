# Custom PySR loss for models/velocity.py — 1-parameter velocity model.
#
# For each candidate DM velocity-shape f, per-galaxy parameters
# (upsilon_disk a, upsilon_bulge b, DM amplitude c, scale radius d) are found
# by inner optimisation:
#   Step 1 – outer loop: for each galaxy g, sweep d on a log grid.
#   Step 2 – inner step: given d, evaluate f(r/d) then solve IRLS for a, b, c.
#   Step 3 – return total Student-t NLL at the best d for each galaxy.
#
# IRLS for Student-t(ν): w_irls_i = w_i·(ν+1)/(ν + w_i·r_i²), r_i = V²_pred − V²_obs.
# Student-t NLL: Σ log(1 + w_i·r_i²/ν) / n_g.
# `tree.trees.f` accesses the f sub-expression of the TemplateExpression,
# letting us evaluate f at custom inputs (r/d) independently of the template.
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

    # ---- physics penalty test points (f(x) independent of d; evaluated once) ----
    # Per-galaxy c amplitude applied at penalty time below.
    X_phys = zeros(T, 5, 7)
    X_phys[1, :] .= T[0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
    f_phys, ok_phys = eval_tree_array(tree.trees.f, X_phys, options)

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
        resid_g  = y_g .- Vgas2_g   # target: a·Vd² + b·Vb² + c·f

        # ---- pre-allocate work arrays ONCE per galaxy (outside d-loop) ----
        X_eval = Matrix{T}(undef, 5, n_g)
        X_eval[2, :] .= T(g)
        X_eval[3, :]  = Vgas2_g
        X_eval[4, :]  = Vdisk2_g
        X_eval[5, :]  = Vbul2_g

        A           = Matrix{T}(undef, n_g, 3)   # design matrix [Vd², Vb², f]
        A_w         = Matrix{T}(undef, n_g, 3)   # sqrt(w)-weighted (initial WLS)
        A_irls      = Matrix{T}(undef, n_g, 3)   # IRLS-weighted design matrix
        r_work      = Vector{T}(undef, n_g)       # residual work vector
        w_irls      = Vector{T}(undef, n_g)       # Student-t IRLS weights
        W_irls_sqrt = Vector{T}(undef, n_g)       # sqrt(w_irls)
        resid_irls  = Vector{T}(undef, n_g)       # IRLS-weighted residual

        A[:, 1]   = Vdisk2_g
        A[:, 2]   = Vbul2_g
        W_sqrt    = sqrt.(w_g)
        resid_w   = W_sqrt .* resid_g
        A_w[:, 1] = W_sqrt .* Vdisk2_g
        A_w[:, 2] = W_sqrt .* Vbul2_g

        # ---- per-galaxy adaptive d range ----
        r_max_g    = maximum(r_g)
        d_min_g    = r_max_g / T(20)
        d_max_g    = max(r_max_g * T(10), T(50))
        log_d_lo_g = log(d_min_g)
        log_d_hi_g = log(d_max_g)

        best_g     = L(Inf)
        best_log_d = (log_d_lo_g + log_d_hi_g) / T(2)

        # Ref for caching IRLS c (params[3]) at the best d seen so far.
        best_loss_seen_g = Ref(L(Inf))
        best_c_g_cache   = Ref(L(1))

        # ---- inner evaluator: WLS→IRLS→Student-t loss at a given log_d ----
        # Captures pre-allocated work arrays from enclosing galaxy scope.
        function eval_candidate_f(log_d_val, n_irls_local::Int=Int(__N_IRLS__))
            d_val = exp(log_d_val)
            X_eval[1, :] .= r_g ./ d_val
            f_vals, flag = eval_tree_array(tree.trees.f, X_eval, options)
            !flag && return L(Inf)
            any(!isfinite, f_vals) && return L(Inf)
            A[:, 3]   .= f_vals
            A_w[:, 3] .= W_sqrt .* f_vals
            maximum(abs, A) < T(1e-10) && return L(Inf)
            params = try
                A_w \ resid_w
            catch
                return L(Inf)
            end
            params = max.(params, L(0))
            for _ in 1:n_irls_local
                r_work .= A * params
                r_work .-= resid_g
                w_irls .= w_g .* ν1 ./ (ν .+ w_g .* r_work .^ 2)
                W_irls_sqrt .= sqrt.(w_irls)
                A_irls[:, 1] .= W_irls_sqrt .* Vdisk2_g
                A_irls[:, 2] .= W_irls_sqrt .* Vbul2_g
                A_irls[:, 3] .= W_irls_sqrt .* f_vals
                resid_irls .= W_irls_sqrt .* resid_g
                new_params = try
                    A_irls \ resid_irls
                catch
                    break
                end
                params = max.(new_params, L(0))
            end
            r_work .= A * params
            r_work .-= resid_g
            st_loss_g = sum(log.(L(1) .+ w_g .* r_work .^ 2 ./ ν)) / n_g
            (isinf(st_loss_g) || isnan(st_loss_g)) && return L(Inf)
            a_opt = params[1]; b_opt = params[2]; c_opt = params[3]
            prior_g = L(1000) * (max(-a_opt, L(0))^2 +
                                 max(-b_opt, L(0))^2 +
                                 max(-c_opt, L(0))^2)
            if upsilon_wt > L(0)
                has_bulge = maximum(abs, Vbul2_g) > T(1e-6)
                if a_opt > L(0)
                    prior_g += upsilon_wt * (log(a_opt) - μ_disk)^2 / (2 * σ_ml^2)
                end
                if has_bulge && b_opt > L(0)
                    prior_g += upsilon_wt * (log(b_opt) - μ_bulge)^2 / (2 * σ_ml^2)
                end
            end
            result_g = st_loss_g + prior_g
            if result_g < best_loss_seen_g[]
                best_loss_seen_g[] = result_g
                best_c_g_cache[]   = c_opt
            end
            return result_g
        end

        # ---- coarse grid: identify basin (full IRLS to match refinement objective) ----
        for log_d in range(log_d_lo_g, log_d_hi_g; length=__N_D_GRID__)
            c = eval_candidate_f(log_d)
            if c < best_g; best_g = c; best_log_d = log_d; end
        end

        # ---- local scan: __N_D_REFINE__-pt uniform over ±1 coarse step around best ----
        d_step_g = (log_d_hi_g - log_d_lo_g) / T(__N_D_GRID__ - 1)
        scan_lo  = max(log_d_lo_g, best_log_d - d_step_g)
        scan_hi  = min(log_d_hi_g, best_log_d + d_step_g)
        for log_d in range(scan_lo, scan_hi; length=__N_D_REFINE__)
            c = eval_candidate_f(log_d)
            if c < best_g; best_g = c; end
        end

        isinf(best_g) && (best_g = L(1e12))
        best_c_g = best_c_g_cache[]
        if ok_phys && all(isfinite, f_phys)
            cf = best_c_g .* f_phys
            origin_pen_g = isfinite(cf[1]) ? min(cf[1]^2, L(1000)) : L(1000)
            nonneg_pen_g = sum(max(-v, L(0))^2 for v in cf)
            xf_g = X_phys[1, :] .* cf
            mono_pen_g = sum(max(xf_g[k] - xf_g[k+1], L(0))^2 for k in 1:6)
            best_g += L(__ORIGIN_WEIGHT__) * origin_pen_g + L(__WEIGHT_NONNEG__) * nonneg_pen_g + L(__WEIGHT_MONO__) * mono_pen_g
        else
            best_g += L(__ORIGIN_WEIGHT__) * L(1000)
        end
        total_loss += best_g
    end

    return total_loss / n_gal
end
