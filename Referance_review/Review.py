import numpy as np
import matplotlib.pyplot as plt

import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx import fem
from dolfinx import mesh,io
from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter,XDMFFile,gmsh as gmshio
from dolfinx.mesh import (locate_entities_boundary,
                            create_submesh)
import ufl
from ufl import (grad,
                 dot,
                 inner,
                 TrialFunction,
                 TestFunction,
                 dx, lhs,
                 nabla_grad,
                 div, rhs,
                 sym)
from dolfinx.fem import (Function, 
                         functionspace,
                         dirichletbc,
                         locate_dofs_topological,
                         form,
                         Constant,
                         extract_function_spaces)
from dolfinx.fem.petsc import (assemble_matrix,
                               assemble_vector,
                               apply_lifting, 
                               create_matrix, 
                               create_vector,
                               set_bc)

from CoolProp.CoolProp import PropsSI
from pathlib import Path

from dolfinx.fem.petsc import create_vector as create_vector_petsc
import inspect
import time
start=time.time()


meshdata=gmshio.read_from_msh('/home/ss/ChemFrontier/Cathode2.msh',MPI.COMM_WORLD,0,2)
domain=meshdata.mesh

cell_tags=meshdata.cell_tags
facet_tags=meshdata.facet_tags

'''
with XDMFFile(MPI.COMM_WORLD, "cathode_domain.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

'''


# Constants
T   = Constant(domain, default_scalar_type(353))
Pc  = Constant(domain, default_scalar_type(1.013*10**5))
R   = Constant(domain, default_scalar_type(80.14))
F   = Constant(domain, default_scalar_type(96487))

MO2      = Constant(domain, default_scalar_type(0.032))
MH2O     = Constant(domain, default_scalar_type(0.018))
DgO2     = Constant(domain, default_scalar_type(1.805*10**-5))
alp      = Constant(domain, default_scalar_type(0.5))
rhowater = Constant(domain, default_scalar_type(974.85))
nuwater  = Constant(domain, default_scalar_type(3.65*10**-7))
K        = Constant(domain, default_scalar_type(5*10**-13))
sigma    = Constant(domain, default_scalar_type(0.0625))
alpc     = Constant(domain, default_scalar_type(1))
epsilon  = Constant(domain,default_scalar_type(0.5))

dH = Constant(domain, default_scalar_type(3e-4))
H1 = Constant(domain, default_scalar_type(1e-3))
L  = Constant(domain, default_scalar_type(0.05))

nuAir=Constant(domain,default_scalar_type(2.1e-5))
muAir=Constant(domain,default_scalar_type(2.09e-5))

# functionspace...\

fdim = domain.topology.dim - 1

import basix.ufl

P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = functionspace(domain, TH)

V, V_to_W = W.sub(0).collapse()   # 속도 서브스페이스 (BC용)
Q, Q_to_W = W.sub(1).collapse()   # 압력 서브스페이스 (필요시)

w = Function(W)          # 이게 (u, p) 둘 다 담는 미지수
u, p = ufl.split(w)      # weak form 쓸 때 이렇게 분리해서 사용
v, q = ufl.TestFunctions(W)   # 테스트함수도 동시에 뽑음

Q0=functionspace(domain,("DG",0))

inlettag=facet_tags.find(11)
walltag=facet_tags.find(14)
outlettag=facet_tags.find(16)
GDLlefttag=facet_tags.find(15)
GDLrighttag=facet_tags.find(17)
GDLtag=cell_tags.find(9)
channeltag=cell_tags.find(8)
uptag=facet_tags.find(13)

W0 = W.sub(0)
inlet_dofs=locate_dofs_topological((W0,V),fdim,inlettag)
wall_dofs=locate_dofs_topological((W0,V),fdim,walltag)
outlet_dofs=locate_dofs_topological((W.sub(1),Q),fdim,outlettag)
GDLleft_dofs=locate_dofs_topological((W0,V),fdim,GDLlefttag)
GDLright_dofs=locate_dofs_topological((W0,V),fdim,GDLrighttag)
up_dofs=locate_dofs_topological((W0,V),fdim,uptag)

darcy_switch=Function(Q0)
darcy_switch.x.array[GDLtag]=1
darcy_switch.x.array[channeltag]=0


u_in = Function(V)
u_in.interpolate(lambda x: np.vstack((np.full(x.shape[1], 0.4), np.zeros(x.shape[1]))))
bc_inlet = dirichletbc(u_in, inlet_dofs, W0)

u_wall = Function(V)   # 기본값 전부 0 (no-slip)
P_out=Function(Q)
bc_wall = dirichletbc(u_wall, wall_dofs, W0)
bc_outlet=dirichletbc(P_out,outlet_dofs,W.sub(1))
bc_GDL=dirichletbc(u_wall,GDLleft_dofs,W0)
bc_GDLr=dirichletbc(u_wall,GDLright_dofs,W0)
bc_up=dirichletbc(u_wall,up_dofs,W0)

bcs = [bc_inlet, bc_wall,bc_outlet,bc_GDL,bc_GDLr,bc_up]

Fres = (
      dot(dot(u, nabla_grad(u)), v) * dx        # 대류항
    + nuAir * inner(grad(u), grad(v)) * dx      # 점성항
    - p * div(v) * dx                           # 압력-속도 커플링
    + q * div(u) * dx                           # 연속방정식 (필수, 이전에 빠졌던 부분)
    + darcy_switch * (muAir/K) * dot(u, v) * dx # GDL Darcy 저항 소스텀
)

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

problem = NonlinearProblem(
    Fres, w, bcs=bcs,
    petsc_options_prefix="ns_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)

problem.solve()
converged_reason = problem.solver.getConvergedReason()
print(f"Converged reason: {converged_reason}")

from dolfinx.io import VTXWriter

u_sol = w.sub(0).collapse()
p_sol = w.sub(1).collapse()

u_sol.name = "velocity"
p_sol.name = "pressure"


with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/velocity13.bp", [u_sol]) as vtx:
    vtx.write(0.0)

with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/pressure13.bp", [p_sol]) as vtx:
    vtx.write(0.0)



# ============================================================
# 종보존 (포화도 s + 산소농도 C) — u_sol, p_sol 계산 직후부터 이어짐
# ============================================================

I_curr = Constant(domain, default_scalar_type(14000))   # A/m^2, 고정 전류밀도
Pv_sat = Constant(domain, default_scalar_type(47400))   # Pa, 353K 물 포화증기압
Mair   = Constant(domain, default_scalar_type(0.029))   # kg/mol

rhoAir = Pc*MO2/(R*T)   # (기존 정의 유지)

ds_measure = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

P_s = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
P_C = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
SC  = basix.ufl.mixed_element([P_s, P_C])
W2  = functionspace(domain, SC)

S_space, _ = W2.sub(0).collapse()
C_space, _ = W2.sub(1).collapse()

# ---- BC (theta 무관, 한 번만) ----
inlet_C_dofs = locate_dofs_topological((W2.sub(1), C_space), fdim, inlettag)
CO2_in = Function(C_space)
CO2_in.interpolate(lambda x: np.full(x.shape[1], 0.21))
bcO2_inlet = dirichletbc(CO2_in, inlet_C_dofs, W2.sub(1))

interface_tag = facet_tags.find(12)   # 채널-GDL 경계, 식(34): s=0
interface_s_dofs = locate_dofs_topological((W2.sub(0), S_space), fdim, interface_tag)
s_zero = Function(S_space)
bcs_interface = dirichletbc(s_zero, interface_s_dofs, W2.sub(0))

bcsO2 = [bcO2_inlet, bcs_interface]

# ---- theta별 반복 계산 ----
results = {}

for theta in [91, 100, 110, 120]:
    theta_rad = np.deg2rad(theta)

    sc = Function(W2)
    sc.sub(0).interpolate(lambda x: np.full(x.shape[1], 0.05))
    sc.sub(1).interpolate(lambda x: np.full(x.shape[1], 0.15))
    sc.x.scatter_forward()

    s, C_O2 = ufl.split(sc)
    vs, v_CO2 = ufl.TestFunctions(W2)

    # ---- 물성함수 (s에 의존) ----
    krl = s**3
    krg = (1 - s)**3
    nu_mix   = 1 / (krl/nuwater + krg/nuAir)
    lambda_l = (krl/nuwater) * nu_mix
    lambda_g = (krg/nuAir)   * nu_mix

    rv_ratio  = (Pv_sat*MH2O) / (Pc*Mair)          # 식(21)
    gamma_H2O = lambda_l + lambda_g*rv_ratio        # 식(24)
    gamma_O2  = 1.0                                 # 산소는 액상에 안 녹음 (식28)

    dJds  = 1.417 - 4.24*s + 3.789*s**2             # dJ(s)/ds
    D_cap = (K*lambda_l*lambda_g/nu_mix) * sigma*np.cos(theta_rad) * (epsilon/K)**0.5 * dJds

    # ---- C_H2O를 s의 대수함수로 계산 (식 6, 21, 22) ----
    rho_mix = rhowater*s + rhoAir*(1-s)             # 식(5)
    C_H2O   = (rhowater*s + rhoAir*(1-s)*rv_ratio) / rho_mix   # 식(6)+(21)+(22)

    # ---- CL 계면(태그13) 반응 flux ----
    water_gen_flux = -I_curr/(2*F)*(1+2*alp)*MH2O     # 식(25)/(26) 기반, 물 생성
    O2_consum_flux = (I_curr/(4*F))*MO2                # 식(35) 기반, 산소 소모

    # ---- F_s: 식(23) 기반 발산형 ----
    F_s = (
          rho_mix * dot(gamma_H2O*u_sol, grad(C_H2O)) * vs * dx
        + (1 - rv_ratio) * D_cap * dot(grad(s), grad(vs)) * dx
        - water_gen_flux * vs * ds_measure(13)
    )

    # ---- F_C: 식(29) 기반 발산형 ----
    F_C = (
          rhoAir * dot(gamma_O2*u_sol, grad(C_O2)) * v_CO2 * dx
        + epsilon*rhoAir*DgO2*(1-s) * dot(grad(C_O2), grad(v_CO2)) * dx
        - D_cap * C_O2 * dot(grad(s), grad(v_CO2)) * dx
        - O2_consum_flux * v_CO2 * ds_measure(13)
    )

    F_total = F_s + F_C

    problem2 = NonlinearProblem(
        F_total, sc, bcs=bcsO2,
        petsc_options_prefix=f"o2sat_{theta}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    problem2.solve()

    reason = problem2.solver.getConvergedReason()
    iters  = problem2.solver.getIterationNumber()
    print(f"theta={theta}: converged reason={reason}, iterations={iters}")

    s_sol = sc.sub(0).collapse()
    C_sol = sc.sub(1).collapse()
    s_sol.name = f"saturation_theta{theta}"
    C_sol.name = f"O2_conc_theta{theta}"

    results[theta] = (s_sol, C_sol)

    print(f"  s range: [{s_sol.x.array.min():.4f}, {s_sol.x.array.max():.4f}]")
    print(f"  C range: [{C_sol.x.array.min():.4f}, {C_sol.x.array.max():.4f}]")


    if theta==91:
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/sss912.bp", [s_sol]) as vtx:
            vtx.write(0.0)
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/CCC912.bp", [C_sol]) as vtx:
            vtx.write(0.0)


    elif theta==100:
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/sss1002.bp", [s_sol]) as vtx:
            vtx.write(0.0)
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/CCC1002.bp", [C_sol]) as vtx:
            vtx.write(0.0)

    elif theta==110:
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/sss1102.bp", [s_sol]) as vtx:
            vtx.write(0.0)
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/CCC1102.bp", [C_sol]) as vtx:
            vtx.write(0.0)

    else:
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/sss1202.bp", [s_sol]) as vtx:
            vtx.write(0.0)
        with VTXWriter(domain.comm, "/home/ss/ChemFrontier/results/CCC1202.bp", [C_sol]) as vtx:
            vtx.write(0.0)


H1_val = 1e-3
dH_val = 3e-4
L_val  = 0.05

n_y = 60      # y방향(GDL 두께) 샘플 개수
n_x = 80      # x방향(채널길이) 평균낼 샘플 개수

y_coords = np.linspace(H1_val, H1_val + dH_val, n_y)
x_coords = np.linspace(0.0005, L_val - 0.0005, n_x)  # 양끝 살짝 안쪽

from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

tree = bb_tree(domain, domain.topology.dim)

def sample_and_average(field, y_coords, x_coords):
    """각 y값마다 x방향으로 평균낸 1D 배열 반환"""
    eta_list = []
    avg_list = []
    for y in y_coords:
        points = np.array([[x, y, 0.0] for x in x_coords])
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(domain, cell_candidates, points)

        vals = []
        for i, point in enumerate(points):
            links = colliding_cells.links(i)
            if len(links) > 0:
                v = field.eval(point.reshape(1,3), [links[0]])
                vals.append(v.flatten()[0])

        if len(vals) > 0:
            avg_list.append(np.mean(vals))
            eta_list.append((y - H1_val) / dH_val)

    return np.array(eta_list), np.array(avg_list)

C_in_val = 0.21

fig1, ax1 = plt.subplots(figsize=(6,5))
fig2, ax2 = plt.subplots(figsize=(6,5))

for theta, (s_sol, C_sol) in results.items():
    eta_s, s_avg = sample_and_average(s_sol, y_coords, x_coords)
    eta_C, C_avg = sample_and_average(C_sol, y_coords, x_coords)

    ax1.plot(eta_s, s_avg, label=f"θc={theta}°")
    ax2.plot(eta_C, C_avg / C_in_val, label=f"θc={theta}°")

ax1.set_xlabel(r"$(y-H_1)/(H_2-H_1)$")
ax1.set_ylabel(r"$s$ (xaverage)")
ax1.set_title("Liquid water saturation across GDL (x-averaged)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.set_xlabel(r"$(y-H_1)/(H_2-H_1)$")
ax2.set_ylabel(r"$C_{O_2,g}/C_{O_2,g,in}$ (x average)")
ax2.set_title("Oxygen concentration across GDL (x-averaged)")
ax2.legend()
ax2.grid(alpha=0.3)

fig1.savefig("/home/ss/ChemFrontier/results/fig2_saturation_avg2.png", dpi=150, bbox_inches="tight")
fig2.savefig("/home/ss/ChemFrontier/results/fig3_oxygen_avg2.png", dpi=150, bbox_inches="tight")

print("그래프 저장 완료: fig2_saturation_avg.png, fig3_oxygen_avg.png")

















end=time.time()
print(f'runtime{end-start}')
