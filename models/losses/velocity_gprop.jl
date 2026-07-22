# Custom PySR loss for models/velocity_gprop.py — joint SR of f(x, γ) and
# γ = g(galaxy properties), v3.
#
# Population model: γ_i ~ LogNormal(g(obs_i), σ_int). Per galaxy the inner
# optimiser maximises likelihood × prior: γ confined to γ_pred·10^{±3σ_int}
# with the smooth penalty ((log10 γ − log10 γ_pred)/σ_int)² added to the NLL
# (negative log of the lognormal prior; replaces the v2 hard band).
#
# Inner optimisation (settings from analysis/diagnose_inner_opt.py, accuracy-
# gated: v2's 5×3 grid + margin-shrunk NM box misranked candidates, Spearman
# 0.83 → 1.00 after these fixes; this config keeps Spearman 1.00 at 2.6× the
# throughput of the maximal one):
#   - coarse grid: __N_START_D__ log-d × __N_START_GAMMA__ γ points at
#     __COARSE_IRLS__ IRLS iterations (cheap basin finding)
#   - Nelder-Mead restarts from the TOP-2 grid cells (multimodal surface),
#     full __N_IRLS__-IRLS objective; best coarse cell rescored at full IRLS
#   - NM bounds = FULL (d, γ) box — no interior margin (edge optima reachable)
#
# Constant-g guard: γ_pred is evaluated for every galaxy up front; if the std
# of the (clamped) predictions across galaxies falls below GVAR_MIN, a large
# smooth penalty (up to WEIGHT_GVAR) is added — g must genuinely vary with
# galaxy properties. (PySR-side constants are disabled — weight_optimize=0,
# complexity_of_constants=99 — rationals come from the one/two/three atoms;
# the guard is what stops g collapsing to a constant expression.)
#
# dataset.X rows: 1=x 2=γ 3=xpow 4=one 5=two 6=three 7=galaxy_code
#                 8=Vgas² 9=Vdisk² 10=Vbulge² 11..(10+N_PROPS)=g's property
#                 features (anchored logs; constant within a galaxy block)
#
# Hyperparameters appear as double-underscored UPPERCASE placeholder tokens,
# filled in by models/julia_losses.py.
using Optim, LinearAlgebra
function physicsloss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # Sterile-member guard: out-of-arity features read zero rows, can score
    # well (atan(x10)=0 is a free zero) yet all children are rejected by
    # check_constraints → sterile tournament winners freeze evolution. Same
    # for real constants while complexity_of_constants=99 bans them (any
    # constant-bearing child exceeds maxsize). Hard-fail both.
    bad_f = any(n -> n.degree == 0 && (n.constant || n.feature > 6), tree.trees.f.tree)
    bad_g = any(n -> n.degree == 0 && (n.constant || n.feature > __G_MAX_FEATURE__), tree.trees.g.tree)
    (bad_f || bad_g) && return L(1e12)

    n_pts      = length(dataset.y)
    n_gal      = Int(maximum(dataset.X[7, :]))
    total_loss = L(0)

    ν  = L(__NU_T__)

    upsilon_wt = L(__UPSILON_WEIGHT__)
    σ_ml       = L(0.1 * log(10))
    μ_disk     = log(L(0.5))
    μ_bulge    = log(L(0.7))

    gamma_lo   = T(__GAMMA_LO__)
    gamma_hi   = T(__GAMMA_HI__)
    σ_int      = T(__SIGMA_INT__)    # lognormal scatter of γ about g(obs), dex
    gr_wt      = L(__WEIGHT_GAMMA_RANGE__)

    n_props  = __N_PROPS__       # galaxy-property features (g's locals 1..n_props)
    n_feat   = __N_FEATURES__    # total dataset features = rows of eval matrices
    gvar_min = T(__GVAR_MIN__)   # min std(γ_pred) across galaxies
    gvar_wt  = L(__WEIGHT_GVAR__)

    three_sigma = T(3) * σ_int   # γ box half-width in dex

    # f eval matrices: f's LOCAL indices 1=x,2=gamma,3=xpow,4=one,5=two,6=three.
    # n_feat rows (= n dataset features): random init trees may reference any
    # feature before check_constraints prunes them; remaining rows stay zero.
    X_origin = zeros(T, n_feat, 1)
    X_origin[1, 1] = T(__ORIGIN_X__)
    X_origin[4, 1] = T(1)
    X_origin[5, 1] = T(2)
    X_origin[6, 1] = T(3)

    n_phys_vel = 6
    X_phys_vel = zeros(T, n_feat, n_phys_vel)
    X_phys_vel[1, :] = [T(0.01), T(0.1), T(0.5), T(1.0), T(3.0), T(10.0)]
    X_phys_vel[4, :] .= T(1)
    X_phys_vel[5, :] .= T(2)
    X_phys_vel[6, :] .= T(3)

    # g eval matrix: g's LOCAL indices 1..n_props = properties, then
    # n_props+1=one, n_props+2=two, n_props+3=three. n_feat rows as above.
    X_gp = zeros(T, n_feat, 1)
    X_gp[n_props + 1, 1] = T(1)
    X_gp[n_props + 2, 1] = T(2)
    X_gp[n_props + 3, 1] = T(3)

    # O(n_pts) galaxy index scan (data sorted by galaxy_code in Python).
    g_start = zeros(Int, n_gal)
    g_end   = zeros(Int, n_gal)
    for i in 1:n_pts
        g = Int(dataset.X[7, i])
        if g_start[g] == 0; g_start[g] = i; end
        g_end[g] = i
    end

    # ---- pre-pass: γ_pred = g(props) for every galaxy + constant-g guard ----
    γ_pred = fill(T(NaN), n_gal)
    for g in 1:n_gal
        g_start[g] == 0 && continue
        for p in 1:n_props
            X_gp[p, 1] = dataset.X[10 + p, g_start[g]]
        end
        g_vals, ok_g = eval_tree_array(tree.trees.g, X_gp, options)
        (!ok_g || !isfinite(g_vals[1])) && return L(1e12)
        γ_pred[g] = g_vals[1]
    end
    # std over CLAMPED predictions: g's that vary wildly but clamp to a
    # constant (e.g. 10·x_M → all γ_c = γ_hi) count as constant too.
    γc_all = [clamp(γ_pred[g], gamma_lo + T(1e-3), gamma_hi - T(1e-3))
              for g in 1:n_gal if g_start[g] != 0]
    n_ok = length(γc_all)
    n_ok == 0 && return L(1e12)
    μ_γ = sum(γc_all) / n_ok
    σ_γ = sqrt(sum(abs2, γc_all .- μ_γ) / n_ok)
    gvar_pen = σ_γ < gvar_min ? gvar_wt * L(1 - σ_γ / gvar_min)^2 : L(0)

    for g in 1:n_gal
        g_start[g] == 0 && continue
        n_g = g_end[g] - g_start[g] + 1
        n_g < 3 && continue

        # ---- γ prior from galaxy properties (pre-pass above) ----
        γ_raw = γ_pred[g]
        γ_c   = clamp(γ_raw, gamma_lo + T(1e-3), gamma_hi - T(1e-3))
        range_pen = min(gr_wt * L(γ_raw - γ_c)^2, L(1e6))
        lg_γc = log10(γ_c)

        # γ box: γ_c·10^{±3σ_int} ∩ (gamma_lo, gamma_hi); prior penalty inside
        # loss_g keeps the box interior soft (max penalty at edge = 9).
        γ_box_lo = max(gamma_lo + T(1e-3), γ_c * exp10(-three_sigma))
        γ_box_hi = min(gamma_hi - T(1e-3), γ_c * exp10(three_sigma))

        rng_g    = g_start[g]:g_end[g]
        r_g      = view(dataset.X, 1, rng_g)
        Vgas2_g  = view(dataset.X, 8, rng_g)
        Vdisk2_g = view(dataset.X, 9, rng_g)
        Vbul2_g  = view(dataset.X, 10, rng_g)
        y_g      = view(dataset.y, rng_g)
        w_g      = view(dataset.weights, rng_g)
        resid_g  = y_g .- Vgas2_g

        W_sqrt  = sqrt.(w_g)
        resid_w = W_sqrt .* resid_g

        has_bulge_g = maximum(abs, Vbul2_g) > T(1e-6)

        # Per-galaxy adaptive d range.
        r_max_g    = maximum(r_g)
        d_min_g    = r_max_g / T(20)
        d_max_g    = max(r_max_g * T(10), T(50))
        log_d_lo_g = log(d_min_g)
        log_d_hi_g = log(d_max_g)

        best_g     = L(Inf)
        best_log_d = (log_d_lo_g + log_d_hi_g) / T(2)
        best_gamma = γ_c

        # ---- Pre-allocate work arrays once per galaxy (outside loss_g). ----
        x_vals      = Vector{T}(undef, n_g)
        xpow_buf    = Vector{T}(undef, n_g)
        X_loc       = zeros(T, n_feat, n_g)
        A           = Matrix{T}(undef, n_g, 3)
        A_w         = Matrix{T}(undef, n_g, 3)
        A_irls      = Matrix{T}(undef, n_g, 3)
        r_work      = Vector{T}(undef, n_g)
        w_irls_buf  = Vector{T}(undef, n_g)
        W_irls_sqrt = Vector{T}(undef, n_g)
        resid_irls  = Vector{T}(undef, n_g)
        coefs_buf   = zeros(T, 3)
        best_coefs  = zeros(T, 3)

        # Fixed columns: f's local indices 4=one, 5=two, 6=three.
        X_loc[4, :] .= T(1)  # one
        X_loc[5, :] .= T(2)  # two
        X_loc[6, :] .= T(3)  # three
        A[:, 1]      = Vdisk2_g
        A[:, 2]      = Vbul2_g
        A_w[:, 1]    = W_sqrt .* Vdisk2_g
        A_w[:, 2]    = W_sqrt .* Vbul2_g

        # ---- Loss closure: NLL + γ prior; mutates preallocated buffers. ----
        function loss_g(params::AbstractVector{T}, n_irls_local::Int=__N_IRLS__)
            log_d_p, γ_p = params[1], params[2]
            d_val = exp(log_d_p)
            @. x_vals   = r_g / d_val
            @. xpow_buf = clamp(x_vals, T(1e-3), T(1e3)) ^ (-γ_p)
            X_loc[1, :] .= x_vals    # x     (local index 1)
            X_loc[2, :] .= γ_p      # gamma (local index 2)
            X_loc[3, :] .= xpow_buf # xpow  (local index 3)

            f_vals, flag = eval_tree_array(tree.trees.f, X_loc, options)
            !flag && return L(1e12)
            any(!isfinite, f_vals) && return L(1e12)
            maximum(abs, f_vals) < T(1e-6) && return L(1e12)

            A[:, 3]   .= f_vals
            A_w[:, 3] .= W_sqrt .* f_vals
            maximum(abs, A) < T(1e-10) && return L(1e12)

            coefs = try; A_w \ resid_w; catch; return L(1e12); end
            coefs = max.(coefs, zero(T))

            for irls_iter in 1:n_irls_local
                r_work .= A * coefs
                r_work .-= resid_g
                @. w_irls_buf = w_g * (ν + one(ν)) / (ν + w_g * r_work ^ 2)
                W_irls_sqrt .= sqrt.(w_irls_buf)
                A_irls[:, 1] .= W_irls_sqrt .* Vdisk2_g
                A_irls[:, 2] .= W_irls_sqrt .* Vbul2_g
                A_irls[:, 3] .= W_irls_sqrt .* f_vals
                resid_irls   .= W_irls_sqrt .* resid_g
                new_coefs = try; A_irls \ resid_irls; catch; break; end
                any(!isfinite, new_coefs) && break
                coefs = max.(new_coefs, zero(T))
            end

            r_work .= A * coefs
            r_work .-= resid_g
            st = sum(log(T(1) + w_g[i] * r_work[i]^2 / ν) for i in eachindex(r_work)) / n_g
            (isinf(st) || isnan(st)) && return L(1e12)

            hinge = T(1000) * (max(-coefs[1], zero(T))^2 +
                               max(-coefs[2], zero(T))^2 +
                               max(-coefs[3], zero(T))^2)
            prior = zero(T)
            if coefs[1] > T(1e-30)
                prior += (log(coefs[1]) - μ_disk)^2 / (2 * σ_ml^2)
            end
            if has_bulge_g && coefs[2] > T(1e-30)
                prior += (log(coefs[2]) - μ_bulge)^2 / (2 * σ_ml^2)
            end
            # lognormal γ prior: negative log of LogNormal(γ_c, σ_int)
            prior_γ = ((log10(γ_p) - lg_γc) / σ_int)^2
            coefs_buf .= coefs
            return st + hinge + upsilon_wt * prior + prior_γ
        end

        # ---- Coarse grid at __COARSE_IRLS__ IRLS (cheap basin find), top-2 ----
        v_s1 = L(Inf); d_s1 = best_log_d; g_s1 = γ_c
        v_s2 = L(Inf); d_s2 = best_log_d; g_s2 = γ_c
        for gamma_j in range(γ_box_lo, γ_box_hi; length=__N_START_GAMMA__)
            for log_d_j in range(log_d_lo_g, log_d_hi_g; length=__N_START_D__)
                c = loss_g(T[log_d_j, gamma_j], __COARSE_IRLS__)
                if c < v_s1
                    v_s2 = v_s1; d_s2 = d_s1; g_s2 = g_s1
                    v_s1 = c;    d_s1 = log_d_j; g_s1 = gamma_j
                elseif c < v_s2
                    v_s2 = c;    d_s2 = log_d_j; g_s2 = gamma_j
                end
            end
        end
        # rescore best coarse cell at the full-IRLS objective
        if isfinite(v_s1)
            best_g = loss_g(T[d_s1, g_s1])
            best_coefs .= coefs_buf
            best_log_d = d_s1
            best_gamma = g_s1
        end

        # ---- Nelder-Mead (Fminbox) restarts from the top-2 coarse cells ----
        # FULL box — no interior margin (v2's d_step/2 shrink excluded edge optima).
        lb_box = T[log_d_lo_g, γ_box_lo]
        ub_box = T[log_d_hi_g, γ_box_hi]
        if isfinite(best_g)
            ϵd = T(1e-6) * (log_d_hi_g - log_d_lo_g)
            ϵg = T(1e-6) * max(γ_box_hi - γ_box_lo, T(1e-6))
            for (d0, g0, v0) in ((d_s1, g_s1, v_s1), (d_s2, g_s2, v_s2))
                isfinite(v0) || continue
                p0_int = T[clamp(d0, lb_box[1] + ϵd, ub_box[1] - ϵd),
                           clamp(g0, lb_box[2] + ϵg, ub_box[2] - ϵg)]
                res = try
                    Optim.optimize(
                        loss_g, lb_box, ub_box, p0_int,
                        Optim.Fminbox(Optim.NelderMead()),
                        Optim.Options(iterations=__OPTIMIZER_NITER__, f_reltol=L(1e-7)))
                catch
                    continue
                end
                v_r = Optim.minimum(res)
                if v_r < best_g
                    best_g     = v_r
                    best_log_d = T(Optim.minimizer(res)[1])
                    best_gamma = T(Optim.minimizer(res)[2])
                    loss_g(T.(Optim.minimizer(res)))
                    best_coefs .= coefs_buf
                end
            end
        end

        # ---- Use c saved at best loss ----
        best_c = best_coefs[3]

        # ---- origin penalty: enforce f(0, γ) ≈ 0 at optimal γ ----
        if L(__ORIGIN_WEIGHT__) > L(0)
            X_origin[2, 1] = best_gamma
            X_origin[3, 1] = T(__ORIGIN_X__) ^ (-best_gamma)
            f_orig, ok_orig = eval_tree_array(tree.trees.f, X_origin, options)
            if ok_orig && isfinite(f_orig[1])
                best_g += L(__ORIGIN_WEIGHT__) * min((best_c * f_orig[1])^2, L(1000))
            else
                best_g += L(__ORIGIN_WEIGHT__) * L(1000)
            end
        end

        # ---- positivity and mass-monotonicity penalties ----
        if L(__WEIGHT_NONNEG__) > L(0) || L(__WEIGHT_MONO__) > L(0)
            X_phys_vel[2, :] .= best_gamma
            X_phys_vel[3, :] .= X_phys_vel[1, :] .^ (-best_gamma)
            f_phys, ok_fphys = eval_tree_array(tree.trees.f, X_phys_vel, options)
            if ok_fphys && all(isfinite, f_phys)
                if L(__WEIGHT_NONNEG__) > L(0)
                    nonneg_pen = sum(max(-(best_c * v), T(0))^2 for v in f_phys)
                    best_g += L(__WEIGHT_NONNEG__) * min(nonneg_pen, L(1e6))
                end
                if L(__WEIGHT_MONO__) > L(0)
                    xf = X_phys_vel[1, :] .* (best_c .* f_phys)
                    mono_pen = sum(max(xf[k] - xf[k+1], T(0))^2 for k in 1:length(xf)-1)
                    best_g += L(__WEIGHT_MONO__) * min(mono_pen, L(1e6))
                end
            end
        end

        isinf(best_g) && (best_g = L(1e12))
        best_g += range_pen
        total_loss += best_g
    end

    return total_loss / n_gal + gvar_pen
end
