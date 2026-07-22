# Custom PySR loss for models/velocity2param.py — f(x, γ) with per-galaxy free γ.
#
# Per galaxy: a, b, c solved analytically (WLS → Student-t IRLS); (log d, γ)
# jointly optimised by a coarse grid then Fminbox Nelder-Mead. Physics
# penalties (origin, non-negativity, mass monotonicity) applied at the optimum.
#
# dataset.X rows: 1=x(r/d) 2=γ 3=xpow=(r/d)^(−γ) 4=one 5=two 6=three
#                 7=galaxy_code 8=Vgas² 9=Vdisk² 10=Vbulge²
#
# Hyperparameters appear as double-underscored UPPERCASE placeholder tokens,
# filled in by models/julia_losses.py.
using Optim, LinearAlgebra
function physicsloss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # Sterile-member guard (same pathology as velocity_gprop.jl): members using
    # features beyond f's arity (>6: galaxy_code, Vgas2, ...) or real constants
    # (banned via complexity_of_constants=99) read zero rows / can never breed
    # valid children under check_constraints, yet can win tournaments and freeze
    # evolution (observed: 115-gal toy run stuck at f = #1 for 40 min).
    any(n -> n.degree == 0 && (n.constant || n.feature > 6), tree.trees.f.tree) && return L(1e12)
    n_pts      = length(dataset.y)
    n_gal      = Int(maximum(dataset.X[7, :]))
    total_loss = L(0)

    ν  = L(__NU_T__)

    upsilon_wt = L(__UPSILON_WEIGHT__)
    σ_ml       = L(0.1 * log(10))
    μ_disk     = log(L(0.5))
    μ_bulge    = log(L(0.7))

    gamma_lo = T(__GAMMA_LO__)
    gamma_hi = T(__GAMMA_HI__)

    # 10-row matrices: global indices 1=x,2=gamma,3=xpow,4=one,5=two,6=three,7-10=aux(zeros)
    X_origin = zeros(T, 10, 1)
    X_origin[1, 1] = T(__ORIGIN_X__)
    X_origin[4, 1] = T(1)
    X_origin[5, 1] = T(2)
    X_origin[6, 1] = T(3)

    n_phys_vel = 6
    X_phys_vel = zeros(T, 10, n_phys_vel)
    X_phys_vel[1, :] = [T(0.01), T(0.1), T(0.5), T(1.0), T(3.0), T(10.0)]
    X_phys_vel[4, :] .= T(1)
    X_phys_vel[5, :] .= T(2)
    X_phys_vel[6, :] .= T(3)

    # O(n_pts) galaxy index scan (data sorted by galaxy_code in Python).
    g_start = zeros(Int, n_gal)
    g_end   = zeros(Int, n_gal)
    for i in 1:n_pts
        g = Int(dataset.X[7, i])
        if g_start[g] == 0; g_start[g] = i; end
        g_end[g] = i
    end

    for g in 1:n_gal
        g_start[g] == 0 && continue
        n_g = g_end[g] - g_start[g] + 1
        n_g < 3 && continue

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
        best_gamma = (gamma_lo + gamma_hi) / T(2)

        # ---- Pre-allocate work arrays once per galaxy (outside loss_g). ----
        x_vals      = Vector{T}(undef, n_g)
        xpow_buf    = Vector{T}(undef, n_g)
        X_loc       = zeros(T, 10, n_g)
        A           = Matrix{T}(undef, n_g, 3)
        A_w         = Matrix{T}(undef, n_g, 3)
        A_irls      = Matrix{T}(undef, n_g, 3)
        r_work      = Vector{T}(undef, n_g)
        w_irls_buf  = Vector{T}(undef, n_g)
        W_irls_sqrt = Vector{T}(undef, n_g)
        resid_irls  = Vector{T}(undef, n_g)
        coefs_buf   = zeros(T, 3)
        best_coefs  = zeros(T, 3)

        # Fixed columns: global indices 4=one, 5=two, 6=three.
        X_loc[4, :] .= T(1)  # one
        X_loc[5, :] .= T(2)  # two
        X_loc[6, :] .= T(3)  # three
        A[:, 1]      = Vdisk2_g
        A[:, 2]      = Vbul2_g
        A_w[:, 1]    = W_sqrt .* Vdisk2_g
        A_w[:, 2]    = W_sqrt .* Vbul2_g

        # ---- Loss closure: mutates preallocated buffers in-place. ----
        function loss_g(params::AbstractVector{T})
            log_d_p, γ_p = params[1], params[2]
            d_val = exp(log_d_p)
            @. x_vals   = r_g / d_val
            @. xpow_buf = clamp(x_vals, T(1e-3), T(1e3)) ^ (-γ_p)
            X_loc[1, :] .= x_vals    # x     (global index 1)
            X_loc[2, :] .= γ_p      # gamma (global index 2)
            X_loc[3, :] .= xpow_buf # xpow  (global index 3)

            f_vals, flag = eval_tree_array(tree.trees.f, X_loc, options)
            !flag && return L(1e12)
            any(!isfinite, f_vals) && return L(1e12)
            maximum(abs, f_vals) < T(1e-6) && return L(1e12)

            A[:, 3]   .= f_vals
            A_w[:, 3] .= W_sqrt .* f_vals
            maximum(abs, A) < T(1e-10) && return L(1e12)

            coefs = try; A_w \ resid_w; catch; return L(1e12); end
            coefs = max.(coefs, zero(T))

            for irls_iter in 1:__N_IRLS__
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
            coefs_buf .= coefs
            return st + hinge + upsilon_wt * prior
        end

        # ---- Coarse __N_START_D__×__N_START_GAMMA__ grid: find best warm start ----
        best_p0 = T[best_log_d, best_gamma]
        for gamma_j in range(gamma_lo, gamma_hi; length=__N_START_GAMMA__+2)[2:end-1]
            for log_d_j in range(log_d_lo_g, log_d_hi_g; length=__N_START_D__+2)[2:end-1]
                c = loss_g(T[log_d_j, gamma_j])
                if c < best_g
                    best_g   = c
                    best_p0  = T[log_d_j, gamma_j]
                    best_coefs .= coefs_buf
                end
            end
        end

        # Sync best_log_d/best_gamma from grid winner before Nelder-Mead and penalties.
        if isfinite(best_g)
            best_log_d = best_p0[1]
            best_gamma = best_p0[2]
        end

        # ---- Nelder-Mead (Fminbox) from best coarse-grid warm start ----
        d_step = (log_d_hi_g - log_d_lo_g) / T(__N_START_D__ + 1)
        g_step = (gamma_hi - gamma_lo)      / T(__N_START_GAMMA__ + 1)
        lb_box = T[log_d_lo_g + d_step / 2, gamma_lo + g_step / 2]
        ub_box = T[log_d_hi_g - d_step / 2, gamma_hi - g_step / 2]
        if isfinite(best_g)
            p0_int = T[clamp(best_p0[1], lb_box[1], ub_box[1]),
                       clamp(best_p0[2], lb_box[2], ub_box[2])]
            res = Optim.optimize(
                loss_g, lb_box, ub_box, p0_int,
                Optim.Fminbox(Optim.NelderMead()),
                Optim.Options(iterations=__OPTIMIZER_NITER__, f_reltol=L(1e-7))
            )
            v_r = Optim.minimum(res)
            if v_r < best_g
                best_g     = v_r
                best_log_d = T(Optim.minimizer(res)[1])
                best_gamma = T(Optim.minimizer(res)[2])
                loss_g(T.(Optim.minimizer(res)))
                best_coefs .= coefs_buf
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
                    best_g += L(__WEIGHT_NONNEG__) * nonneg_pen
                end
                if L(__WEIGHT_MONO__) > L(0)
                    xf = X_phys_vel[1, :] .* (best_c .* f_phys)
                    mono_pen = sum(max(xf[k] - xf[k+1], T(0))^2 for k in 1:length(xf)-1)
                    best_g += L(__WEIGHT_MONO__) * mono_pen
                end
            end
        end

        isinf(best_g) && (best_g = L(1e12))
        total_loss += best_g
    end

    return total_loss / n_gal
end
