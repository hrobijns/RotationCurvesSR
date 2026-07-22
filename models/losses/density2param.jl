# Custom PySR loss for models/density2param.py — density shape h(x, γ) with
# per-galaxy free γ, integrated to a DM column via cumulative trapezoid.
#
# Per galaxy: a, b, c solved analytically (WLS → Student-t IRLS); (log d, γ)
# optimised by a coarse 2-D grid (WLS-only) with top-3 basin tracking, then a
# hand-rolled Nelder-Mead refinement from each of the top-3 starts. Physics
# penalties (non-negativity, inner log-slope 0 ≤ s ≤ 2) applied at the optimum.
#
# X_eval rows: 1=x(r/d) 2=γ 3=xpow=(r/d)^γ (f has inputs r, gamma, xpow).
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
    ν1 = ν + L(1)

    # Log-normal priors on upsilon_disk (a) and upsilon_bulge (b),
    # matching mcmc_fit.py: lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex).
    upsilon_wt = L(__UPSILON_WEIGHT__)
    σ_ml       = L(0.1 * log(10))
    μ_disk     = log(L(0.5))
    μ_bulge    = log(L(0.7))

    # Cuspiness parameter grid (linearly spaced, per-galaxy optimised)
    gamma_lo = T(__GAMMA_LO__)
    gamma_hi = T(__GAMMA_HI__)
    n_gamma  = __N_GAMMA_GRID__

    # Physics penalty weights
    weight_nonneg = L(__WEIGHT_NONNEG__)   # penalise h < 0
    # weight_slope applied inline below   # enforce 0 ≤ inner log-slope ≤ 2

    # Physics test points: x=0.001 at front for inner log-slope check.
    # Row layout: row1=x, row2=γ (set per galaxy), row3=xpow=x^γ (set per galaxy).
    n_phys = 7
    X_phys = zeros(T, 6, n_phys)
    X_phys[1, :] = [T(0.001), T(0.01), T(0.1), T(0.5), T(1.0), T(3.0), T(10.0)]

    # ---- pre-compute per-galaxy index ranges (data sorted by galaxy_code) ----
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
        g_start[g] == 0 && continue
        n_g = g_end[g] - g_start[g] + 1
        n_g < 3 && continue

        rng_g    = g_start[g]:g_end[g]
        r_g      = view(dataset.X, 1, rng_g)
        Vgas2_g  = view(dataset.X, 3, rng_g)
        Vdisk2_g = view(dataset.X, 4, rng_g)
        Vbul2_g  = view(dataset.X, 5, rng_g)
        y_g      = view(dataset.y, rng_g)
        w_g      = view(dataset.weights, rng_g)
        resid_g  = y_g .- Vgas2_g

        # ---- pre-allocate work arrays ONCE per galaxy (outside 2D grid loop) ----
        # Row layout for X_eval (passed to eval_tree_array for SR expression f):
        #   row1 = r/d   (updated per d-iteration inside eval_candidate_g)
        #   row2 = γ     (updated per γ-iteration in the outer loop below)
        #   row3 = xpow  (= (r/d)^γ, updated inside eval_candidate_g)
        # f has 3 positional inputs (r, gamma, xpow) so only rows 1-3 are accessed.
        X_eval = Matrix{T}(undef, 6, n_g)

        A           = Matrix{T}(undef, n_g, 3)
        A_w         = Matrix{T}(undef, n_g, 3)
        A_irls      = Matrix{T}(undef, n_g, 3)
        r_work      = Vector{T}(undef, n_g)
        w_irls      = Vector{T}(undef, n_g)
        W_irls_sqrt = Vector{T}(undef, n_g)
        resid_irls  = Vector{T}(undef, n_g)
        dm_col  = Vector{T}(undef, n_g)
        h_g     = Vector{T}(undef, n_g)   # DM density shape h(x, γ)

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
        best_gamma = (gamma_lo + gamma_hi) / T(2)

        # ---- nested helper: evaluate loss at a given log(d) ----
        # X_eval[2, :] must already hold the current γ value (set by caller).
        # n_irls_local=0 → WLS only (cheap basin-finding on coarse grid);
        # n_irls_local=__N_IRLS__ → full IRLS (used during Nelder-Mead refinement).
        function eval_candidate_g(log_d_val, n_irls_local::Int=Int(__N_IRLS__))
            d_val = exp(log_d_val)
            X_eval[1, :] .= r_g ./ d_val
            X_eval[3, :] .= X_eval[1, :] .^ X_eval[2, 1]  # xpow = (r/d)^γ at row 3
            # X_eval[2, :] holds γ — set by caller before invoking this function.

            # h_g[i] = SR expression evaluated directly as density shape h(x_i, γ).
            # Negative values clamped to 0; penalised at physics test points.
            h_vals, fl = eval_tree_array(tree.trees.f, X_eval, options)
            !fl && return L(Inf)
            any(!isfinite, h_vals) && return L(Inf)
            h_g .= max.(h_vals, T(0))

            cum_I_loc = T(0); prev_hr2 = T(0); prev_r_loc = T(0)
            for i in 1:n_g
                r_i    = r_g[i]
                h_r2_i = h_g[i] * r_i * r_i
                cum_I_loc  += (h_r2_i + prev_hr2) * T(0.5) * (r_i - prev_r_loc)
                dm_col[i]   = cum_I_loc / r_i
                prev_hr2    = h_r2_i
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

        # ---- Nelder-Mead 2D: minimise loss over (log_d, γ) ----
        # Captures X_eval, eval_candidate_g, and per-galaxy bounds from enclosing scope.
        # n_gs_iter controls max iterations (same parameter as the old golden-section).
        function nelder_mead_2d(ld0, ga0, step_d, step_g)
            ld1 = clamp(ld0,          log_d_lo_g, log_d_hi_g)
            ga1 = clamp(ga0,          gamma_lo,   gamma_hi)
            ld2 = clamp(ld0 + step_d, log_d_lo_g, log_d_hi_g)
            ga2 = ga1
            ld3 = ld1
            ga3 = clamp(ga0 + step_g, gamma_lo, gamma_hi)
            X_eval[2, :] .= ga1; v1 = eval_candidate_g(ld1)
            X_eval[2, :] .= ga2; v2 = eval_candidate_g(ld2)
            X_eval[2, :] .= ga3; v3 = eval_candidate_g(ld3)
            for _ in 1:__N_GS_ITER__
                if v2 < v1; ld1,ld2 = ld2,ld1; ga1,ga2 = ga2,ga1; v1,v2 = v2,v1; end
                if v3 < v2; ld2,ld3 = ld3,ld2; ga2,ga3 = ga3,ga2; v2,v3 = v3,v2; end
                if v2 < v1; ld1,ld2 = ld2,ld1; ga1,ga2 = ga2,ga1; v1,v2 = v2,v1; end
                v3 - v1 < L(1e-5) && break
                ldc = (ld1 + ld2) / T(2); gac = (ga1 + ga2) / T(2)
                ldr = clamp(T(2)*ldc - ld3, log_d_lo_g, log_d_hi_g)
                gar = clamp(T(2)*gac - ga3, gamma_lo, gamma_hi)
                X_eval[2, :] .= gar; vr = eval_candidate_g(ldr)
                if vr < v1
                    lde = clamp(T(2)*ldr - ldc, log_d_lo_g, log_d_hi_g)
                    gae = clamp(T(2)*gar - gac, gamma_lo, gamma_hi)
                    X_eval[2, :] .= gae; ve = eval_candidate_g(lde)
                    if ve < vr; ld3,ga3,v3 = lde,gae,ve
                    else;       ld3,ga3,v3 = ldr,gar,vr; end
                elseif vr < v2
                    ld3,ga3,v3 = ldr,gar,vr
                else
                    lds = clamp((ld3 + ldc) / T(2), log_d_lo_g, log_d_hi_g)
                    gas = clamp((ga3 + gac) / T(2), gamma_lo, gamma_hi)
                    X_eval[2, :] .= gas; vs = eval_candidate_g(lds)
                    if vs < v3
                        ld3,ga3,v3 = lds,gas,vs
                    else
                        ld2 = clamp((ld1+ld2)/T(2), log_d_lo_g, log_d_hi_g)
                        ga2 = clamp((ga1+ga2)/T(2), gamma_lo, gamma_hi)
                        X_eval[2, :] .= ga2; v2 = eval_candidate_g(ld2)
                        ld3 = clamp((ld1+ld3)/T(2), log_d_lo_g, log_d_hi_g)
                        ga3 = clamp((ga1+ga3)/T(2), gamma_lo, gamma_hi)
                        X_eval[2, :] .= ga3; v3 = eval_candidate_g(ld3)
                    end
                end
            end
            if v1 <= v2 && v1 <= v3; return ld1, ga1, v1
            elseif v2 <= v3;          return ld2, ga2, v2
            else;                     return ld3, ga3, v3; end
        end   # nelder_mead_2d

        # ---- coarse 2D grid: WLS only (n_irls=0) — top-3 basin tracking ----
        # γ linearly spaced in [gamma_lo, gamma_hi]; d log-spaced in per-galaxy range.
        ld_s1 = best_log_d; ga_s1 = best_gamma; v_s1 = L(Inf)
        ld_s2 = best_log_d; ga_s2 = best_gamma; v_s2 = L(Inf)
        ld_s3 = best_log_d; ga_s3 = best_gamma; v_s3 = L(Inf)

        for j in 1:n_gamma
            gamma_j = n_gamma == 1 ?
                      (gamma_lo + gamma_hi) / T(2) :
                      gamma_lo + (gamma_hi - gamma_lo) * T(j - 1) / T(n_gamma - 1)
            X_eval[2, :] .= gamma_j
            for log_d in range(log_d_lo_g, log_d_hi_g; length=__N_D_GRID__)
                c = eval_candidate_g(log_d, 0)   # WLS only on coarse grid
                if c < v_s1
                    ld_s3,ga_s3,v_s3 = ld_s2,ga_s2,v_s2
                    ld_s2,ga_s2,v_s2 = ld_s1,ga_s1,v_s1
                    ld_s1,ga_s1,v_s1 = log_d, gamma_j, c
                elseif c < v_s2
                    ld_s3,ga_s3,v_s3 = ld_s2,ga_s2,v_s2
                    ld_s2,ga_s2,v_s2 = log_d, gamma_j, c
                elseif c < v_s3
                    ld_s3,ga_s3,v_s3 = log_d, gamma_j, c
                end
            end
        end

        # IRLS warm-start at best coarse grid point
        X_eval[2, :] .= ga_s1; best_g = eval_candidate_g(ld_s1)
        best_log_d = ld_s1; best_gamma = ga_s1

        # ---- Nelder-Mead refinement from top-3 coarse starts ----
        step_d_nm = (log_d_hi_g - log_d_lo_g) / T(__N_D_GRID__)
        step_g_nm = (gamma_hi - gamma_lo) / T(__N_GAMMA_GRID__)
        for (ld0_nm, ga0_nm) in ((ld_s1, ga_s1), (ld_s2, ga_s2), (ld_s3, ga_s3))
            nm_ld, nm_ga, nm_v = nelder_mead_2d(ld0_nm, ga0_nm, step_d_nm, step_g_nm)
            if nm_v < best_g
                best_g = nm_v; best_log_d = nm_ld; best_gamma = nm_ga
            end
        end

        X_eval[2, :] .= best_gamma   # restore for physics penalty

        # ---- Physics penalties at optimal (d, γ) ----
        # Non-negativity: penalise h < 0 at 7 test points.
        # Inner log-slope: enforce 0 ≤ s ≤ 2 at innermost interval [0.001, 0.01].
        if weight_nonneg > L(0) || L(__WEIGHT_SLOPE__) > L(0)
            X_phys[2, :] .= best_gamma
            X_phys[3, :] .= X_phys[1, :] .^ best_gamma   # xpow at row 3
            h_phys, ok_phys = eval_tree_array(tree.trees.f, X_phys, options)
            if ok_phys && all(isfinite, h_phys)
                if weight_nonneg > L(0)
                    best_g += weight_nonneg * sum(max(-v, T(0)) for v in h_phys)
                end
                if L(__WEIGHT_SLOPE__) > L(0) && h_phys[1] > T(1e-8) && h_phys[2] > T(1e-8)
                    s_inner = log(h_phys[1] / h_phys[2]) / log(X_phys[1, 2] / X_phys[1, 1])
                    slope_pen = max(-s_inner, T(0)) + max(s_inner - T(2), T(0))
                    best_g += L(__WEIGHT_SLOPE__) * slope_pen
                end
            end
        end

        isinf(best_g) && (best_g = L(1e12))
        total_loss += best_g
    end

    return total_loss / n_gal
end
