"""Hard-constraint matrix-zonotope RPC with nominal terminal-speed projection.

Nominal directions are preserved during recursive uncertainty propagation.
Support cuts solve the same norm constraints; every result is independently
checked using explicit generator propagation. No slack or fallback is used.
"""
import time
import cvxpy as cp
import numpy as np
import functions as f

def fixed_head_generators(A, H, N, eps):
    heads = [np.zeros((A.shape[0], 0))]
    for _ in range(N):
        heads.append(np.column_stack([A @ heads[-1], H[:, 0] * eps]))
    return heads

def numeric_generator_rollout(bundle, z0, u, ep, eps):
    """Independent generator propagation using the configured compressed matrix sets."""
    (A, B, H, C) = [bundle[k] for k in ['A', 'B', 'H', 'C']]
    (P, Pc) = (bundle.get('D_mz_generators', bundle['D_pinv']), bundle.get('Zp_mz_generators', bundle['Zp_pinv']))
    (bs, br) = (bundle['sigma_max'], bundle['rho_max'])
    z = np.asarray(z0).copy()
    G = np.zeros((len(z), 0))
    out = []
    for (k, value) in enumerate(u):
        center = np.r_[z, value, ep[k]]
        Gin = np.zeros((len(z) + 2, G.shape[1] + 1))
        Gin[:len(z), :G.shape[1]] = G
        Gin[-1, -1] = eps
        r = bs * (np.abs(P @ center).sum() + np.abs(P @ Gin).sum()) + bs
        M = np.hstack([A, B, H])
        z = M @ center
        G = np.column_stack([M @ Gin, np.diag(r)])
        rx = br * (np.abs(Pc @ z).sum() + np.abs(Pc @ G).sum()) + br
        hx = np.abs(C @ G).sum(axis=1) + rx
        out.append({'z': z.copy(), 'x': C @ z, 'hx': hx, 'r': r})
    return out

class RNDDPCSolver:

    def __init__(self,bundle,Nfut,scenario,physical_mode='physical_hard'):
        if physical_mode!='physical_hard':
            raise ValueError('This trial requires hard physical bounds')
        A,B,H,C=[bundle[k] for k in ['A','B','H','C']]
        P,Pc=bundle.get('D_mz_generators',bundle['D_pinv']),bundle.get('Zp_mz_generators',bundle['Zp_pinv'])
        bs,br=bundle['sigma_max'],bundle['rho_max']
        nz,nx=A.shape[0],C.shape[0]
        self.n_z,self.s_dim,self.Nfut=nz,nx,Nfut
        self.z_t=cp.Parameter(nz);self.x0=cp.Parameter(nx);self.u_prev=cp.Parameter(1)
        self.x_ref=cp.Parameter((nx,Nfut+1));self.ep=cp.Parameter((1,Nfut+1))
        self.z=cp.Variable((nz,Nfut+1));self.x=cp.Variable((nx,Nfut+1));self.u=cp.Variable((1,Nfut))
        self.hx=cp.Variable((nx,Nfut),nonneg=True)
        self.injected_radius=cp.Variable((nz,Nfut),nonneg=True)
        self.eta_z=cp.Variable(Nfut,nonneg=True);self.eta_x=cp.Variable(Nfut,nonneg=True)
        powers=[np.linalg.matrix_power(A,k) for k in range(Nfut)]
        heads=fixed_head_generators(A,H,Nfut,f.EPS_MAX)
        cons=[self.z[:,0]==self.z_t,self.x[:,0]==self.x0]
        Q=np.diag([f.rnddpc_q_spa,f.rnddpc_q_vel]*(nx//2));obj=0
        for t in range(Nfut):
            center=cp.hstack([self.z[:,t],self.u[0,t],self.ep[0,t]])
            coeff=np.abs(P[:,:nz]@heads[t]).sum()+f.EPS_MAX*np.abs(P[:,-1]).sum()
            for l in range(t):
                coeff=coeff+np.abs(P[:,:nz]@powers[t-1-l]).sum(axis=0)@self.injected_radius[:,l]
            nominal_radius=np.abs(C@heads[t+1]).sum(axis=1)
            rho_coeff=np.abs(Pc@heads[t+1]).sum()
            for l in range(t+1):
                nominal_radius=nominal_radius+np.abs(C@powers[t-l])@self.injected_radius[:,l]
                rho_coeff=rho_coeff+np.abs(Pc@powers[t-l]).sum(axis=0)@self.injected_radius[:,l]
            cons += [self.eta_z[t]>=cp.norm1(P@center),
                     self.z[:,t+1]==A@self.z[:,t]+B[:,0]*self.u[0,t]+H[:,0]*self.ep[0,t],
                     self.injected_radius[:,t]==cp.multiply(bs,self.eta_z[t]+coeff)+bs,
                     self.x[:,t+1]==C@self.z[:,t+1],self.eta_x[t]>=cp.norm1(Pc@self.z[:,t+1]),
                     self.hx[:,t]==nominal_radius+cp.multiply(br,self.eta_x[t]+rho_coeff)+br,
                     self.x[0,t+1]-self.hx[0,t]>=f.SpaMin,self.x[0,t+1]+self.hx[0,t]<=f.SpaMax,
                     self.x[1,t+1]-self.hx[1,t]>=f.VelMin,self.x[1,t+1]+self.hx[1,t]<=f.VelMax,
                     self.u[0,t]>=f.AccMin,self.u[0,t]<=f.AccMax]
            obj+=cp.quad_form(self.x[:,t+1]-self.x_ref[:,t+1],Q)+f.rnddpc_r*cp.sum_squares(self.u[:,t])
        obj+=f.rnddpc_rd*cp.sum_squares(self.u[:,0]-self.u_prev)
        for t in range(1,Nfut):
            obj+=f.rnddpc_rd*cp.sum_squares(self.u[:,t]-self.u[:,t-1])
        self.constraints_without_terminal=cons
        self.target=cp.Parameter(name='projected_terminal_velocity')
        self.prob=cp.Problem(cp.Minimize(obj),cons+[self.x[1,Nfut]==self.target])
        self.projection_problem=cp.Problem(cp.Minimize(cp.square(self.x[1,Nfut]-self.x_ref[1,Nfut])),cons)
        self.solver_name=f.CONIC_SOLVER;self.last_projection={}
        self.bundle = bundle
        self.P = bundle.get('D_mz_generators', bundle['D_pinv'])
        self.Pc = bundle.get('Zp_mz_generators', bundle['Zp_pinv'])
        self.norms = []
        for k in range(Nfut):
            self.norms.append((self.eta_z[k], self.P, cp.hstack([self.z[:, k], self.u[0, k], self.ep[0, k]])))
            self.norms.append((self.eta_x[k], self.Pc, self.z[:, k + 1]))
        self.base_constraints = [c for c in self.constraints_without_terminal
                                 if not isinstance(c.args[0], cp.norm1)]
        assert len(self.constraints_without_terminal) - len(self.base_constraints) == 2*Nfut
        self.main_objective = self.prob.objective
        self.cuts = []
        self.seen = [set() for _ in self.norms]
        self.seeded = False
        self.rounds = 0
        self.cut_solver_seconds = 0.0
        self._rebuild()

    def _rebuild(self):
        self.constraints_without_terminal = self.base_constraints + self.cuts
        self.prob = cp.Problem(self.main_objective, self.constraints_without_terminal + [self.x[1, self.Nfut] == self.target])
        self.projection_problem = cp.Problem(cp.Minimize(cp.square(self.x[1, self.Nfut] - self.x_ref[1, self.Nfut])), self.constraints_without_terminal)

    def _cut(self, index, point):
        (eta, P, expression) = self.norms[index]
        signs = np.sign(P @ point)
        key = signs.astype(np.int8).tobytes()
        if key in self.seen[index]:
            return False
        self.seen[index].add(key)
        self.cuts.append(eta >= P.T @ signs @ expression)
        return True

    def _seed(self, z0, ep):
        z = np.asarray(z0).copy()
        for k in range(self.Nfut):
            self._cut(2 * k, np.r_[z, 0.0, ep[0, k]])
            z = self.bundle['A'] @ z + self.bundle['H'][:, 0] * ep[0, k]
            self._cut(2 * k + 1, z)
        self._rebuild()
        self.seeded = True

    def _solve_separated(self, projection=False):
        """Add valid support cuts until all original norm constraints hold."""
        for _ in range(100):
            problem = self.projection_problem if projection else self.prob
            problem.solve(solver=self.solver_name, warm_start=True, verbose=False, tol_gap_abs=1e-09, tol_gap_rel=1e-09, tol_feas=1e-09, max_iter=200)
            self.rounds += 1
            if problem.solver_stats.solve_time is not None:
                self.cut_solver_seconds += problem.solver_stats.solve_time
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return problem.status
            added = 0
            largest = 0.0
            for (index, (eta, P, expression)) in enumerate(self.norms):
                point = np.asarray(expression.value).reshape(-1)
                norm = float(np.abs(P @ point).sum())
                gap = norm - float(eta.value)
                largest = max(largest, gap)
                if gap > 1e-08 + 1e-12 * norm:
                    added += int(self._cut(index, point))
            if added:
                self._rebuild()
                continue
            if largest > 1e-05:
                raise RuntimeError(f'Norm separation stalled with residual {largest}')
            self.last_norm_residual = largest
            return problem.status
        raise RuntimeError('Exact norm separation iteration limit reached')

    def solve(self, z_t_val, x0_val, x_ref_val, ep_val, u_prev_val):
        started = time.perf_counter()
        self.z_t.value = z_t_val
        self.x0.value = x0_val
        self.x_ref.value = x_ref_val
        self.ep.value = ep_val
        self.u_prev.value = np.asarray(u_prev_val, dtype=float).reshape(1)
        raw = float(x_ref_val[1, self.Nfut])
        self.target.value = raw
        self.last_projection = {'raw_target': raw, 'projection_used': False}
        self.rounds = 0
        self.cut_solver_seconds = 0.0
        if not self.seeded:
            self._seed(z_t_val, ep_val)
        status = self._solve_separated()
        self.last_projection['raw_problem_status'] = status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.last_projection['projection_used'] = True
            status = self._solve_separated(projection=True)
            self.last_projection['projection_problem_status'] = status
            if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError('No feasible tightened terminal-speed interval: ' + str(status))
            self.target.value = float(self.x.value[1, self.Nfut])
            status = self._solve_separated()
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError('Separated projected control problem failed: ' + str(status))
        reference = numeric_generator_rollout(self.bundle, np.asarray(z_t_val), self.u.value[0], ep_val[0], f.EPS_MAX)
        true_hx = np.stack([r['hx'] for r in reference], axis=1)
        centers = np.stack([r['x'] for r in reference], axis=1)
        np.testing.assert_allclose(self.x.value[:, 1:], centers, rtol=1e-07, atol=1e-06)
        violation = max(float(np.max(f.SpaMin - (centers[0] - true_hx[0]))), float(np.max(centers[0] + true_hx[0] - f.SpaMax)), float(np.max(f.VelMin - (centers[1] - true_hx[1]))), float(np.max(centers[1] + true_hx[1] - f.VelMax)), 0.0)
        if violation > 1e-05:
            raise RuntimeError(f'Independent generator propagation violates a hard bound by {violation}')
        target = float(self.target.value)
        residual = abs(float(self.x.value[1, -1]) - target)
        if residual > 1e-05:
            raise RuntimeError('Nominal terminal equality validation failed')
        self.last_projection.update(projected_target=target, final_lower=f.VelMin + float(true_hx[1, -1]), final_upper=f.VelMax - float(true_hx[1, -1]), equality_residual=residual, separation_rounds=self.rounds, total_support_cuts=len(self.cuts), full_norm_residual=self.last_norm_residual, full_propagation_violation=violation)
        return (self.x.value, self.z.value, self.u.value, time.perf_counter() - started, self.cut_solver_seconds, true_hx)
