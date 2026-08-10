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
R   = Constant(domain, default_scalar_type(8.314))
F   = Constant(domain, default_scalar_type(96487))

MO2      = Constant(domain, default_scalar_type(0.032))
MH2O     = Constant(domain, default_scalar_type(0.018))
DgO2     = Constant(domain, default_scalar_type(1.805*10**-5))
alp      = Constant(domain, default_scalar_type(0.5))
rhowater = Constant(domain, default_scalar_type(974.85))
nuwater  = Constant(domain, default_scalar_type(3.65*10**-7))
muwater = Constant(domain, default_scalar_type(3.558e-4))
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
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1) # pressure
P1_C=basix.ufl.element("Lagrange",domain.basix_cell(),1) # concentration
P1_W=basix.ufl.element("Lagrange",domain.basix_cell(),1) # water concen

TH = basix.ufl.mixed_element([P2, P1,P1_C,P1_W])
W = functionspace(domain, TH)

V, V_to_W = W.sub(0).collapse()   # 속도 서브스페이스 (BC용)
Q, Q_to_W = W.sub(1).collapse()   # 압력 서브스페이스 (필요시)
V_C,VC_to_W=W.sub(2).collapse()
V_W,VW_top_W=W.sub(3).collapse()




w = Function(W)          # 이게 (u, p) 둘 다 담는 미지수
u, p,C,CW = ufl.split(w)      # weak form 쓸 때 이렇게 분리해서 사용
v, q,vC,vW = ufl.TestFunctions(W)   # 테스트함수도 동시에 뽑음

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

# GDL 에서 역류가 일어난다고함.. 일단 자연조건으로..

bc_GDL=dirichletbc(u_wall,GDLleft_dofs,W0)
#bc_GDLr=dirichletbc(u_wall,GDLright_dofs,W0)
bc_up=dirichletbc(u_wall,up_dofs,W0)

bcs = [bc_inlet, bc_wall,bc_outlet,bc_GDL,bc_up]


#유로에서 속도 초기값 0.4로 시작.. 가스확산층은 0으로.
w.sub(0).interpolate(lambda x: np.vstack((np.full(x.shape[1], 0.4), np.zeros(x.shape[1]))),cells0=channeltag)




I_ref = Constant(domain, default_scalar_type(100))   # A/m^2, 고정 전류밀도
eta=Constant(domain,default_scalar_type(0.05)) # 과전압 시작값.
V_oc=Constant(domain,default_scalar_type(1.2)) # 표준환원전위

Pv_sat = Constant(domain, default_scalar_type(47400))   # Pa, 353K 물 포화증기압
Mair   = Constant(domain, default_scalar_type(0.029))   # kg/mol

rhoAir = Pc*MO2/(R*T)   # (기존 정의 유지)

ds_measure = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

inlet_C_dofs = locate_dofs_topological((W.sub(2), V_C), fdim, inlettag)
CO2_in = Function(V_C)
CO2_in.interpolate(lambda x: np.full(x.shape[1], 0.21))
bcO2_inlet = dirichletbc(CO2_in, inlet_C_dofs, W.sub(2))
w.sub(2).interpolate(lambda x: np.full(x.shape[1], 0.21))
w.x.scatter_forward()



inlet_CW_dofs = locate_dofs_topological((W.sub(3), V_W), fdim, inlettag)
CW_in = Function(V_W)
CW_in.interpolate(lambda x: np.full(x.shape[1], 0.01))
bcW_inlet = dirichletbc(CW_in, inlet_CW_dofs, W.sub(3))
w.sub(3).interpolate(lambda x: np.full(x.shape[1], 0.01))
w.x.scatter_forward()

bcsW=[bcW_inlet]

bcsO2 = [bcO2_inlet]

bc_all=bcsO2+bcs+bcsW

theta_rad=np.deg2rad(91) # 접촉각 지정

rho_g_const=(101325*Mair)/(R*T)

Cg_sat = (Pv_sat*MH2O) / (Pv_sat*MH2O + (Pc - Pv_sat)*Mair)   # 포화 수증기 질량분율
s_smooth_eps = 0.02

def s_expr(Cw):
    raw = rho_g_const*(Cw - Cg_sat) / (rhowater*(1 - Cw) + rho_g_const*(Cw - Cg_sat) + 1e-12)
    s_smooth = 0.5*(raw + ufl.sqrt(raw**2 + s_smooth_eps**2))   # smooth max(raw, 0)
    s_smooth = ufl.max_value(s_smooth, 0.0)
    return ufl.min_value(s_smooth, 0.99)                        # smooth min(., s_max)

s_raw = s_expr(CW)
s= ufl.variable(s_raw)





krl = s**3
krg = (1 - s)**3
nu_mix   = 1 / (krl/nuwater + krg/nuAir)
lambda_l = (krl/nuwater) * nu_mix
lambda_g = (krg/nuAir)   * nu_mix

rv_ratio  = (Pv_sat*MH2O) / (Pc*Mair)          # 식(21)
gamma_H2O = lambda_l + lambda_g*rv_ratio        # 식(24)
gamma_O2  = lambda_g                                 # 산소는 액상에 안 녹음 (식28)

dJds  = 1.417 - 4.24*s + 3.789*s**2             # dJ(s)/ds
D_cap = (K*lambda_l*lambda_g/nu_mix) * sigma*np.cos(theta_rad) * (epsilon/K)**0.5 * dJds

DO2=epsilon*(1-s)*DgO2


# ---- C_H2O를 s의 대수함수로 계산 (식 6, 21, 22) ----
rho_mix = rhowater*s + rhoAir*(1-s)             # 식(5)



Fu=(
      dot(dot(u, nabla_grad(u)), v) * dx        # 대류항
    + nuAir * inner(grad(u), grad(v)) * dx      # 점성항
    - p * div(v) * dx                           # 압력-속도 커플링
    + q * div(u) * dx                           # 연속방정식 (필수, 이전에 빠졌던 부분)
    + darcy_switch * (muAir/K) * dot(u, v) * dx # GDL Darcy 저항 소스텀
)



n = ufl.FacetNormal(domain)


# ---- Tafel 전류 (논문 식 3.5) ----
# alpc를 써야 함 (alp=0.5는 anodic 쪽), CO2_in Function 대신 상수 기준농도 사용
CO2_ref = Constant(domain, default_scalar_type(0.21))
I = (1 - s) * I_ref * (C / CO2_ref) * ufl.exp(alpc * F * eta / (R * T))

# ---- 아웃렛 클리핑 (논문 §5.2.1 핵심 stabilization) ----
u_n = dot(u, n)
u_n_out = ufl.conditional(ufl.gt(u_n, 0), u_n, 0)   # max(u·n, 0), 역류 시 0으로 클리핑

# ============ 물/포화도 수송 (식 3.2 구조, s가 1차 미지수) ============
water_diff_flux = D_cap * grad(s)
water_conv_flux = gamma_H2O * rho_mix * u 

DgH2O = Constant(domain, default_scalar_type(2.56e-5))   # 수증기-공기 확산계수, m^2/s

water_diff_flux = epsilon*(1 - s)*DgH2O*grad(CW) + D_cap*grad(s)
FH2O = (
      dot(water_diff_flux, grad(vW)) * dx
    - dot(water_conv_flux, grad(vW)) * dx
    + gamma_H2O * rho_mix  * u_n_out * vW * ds_measure(16)
    - (I / (2 * F)) * MH2O * vW * ds_measure(13)
)
# ============ 산소 수송 (식 3.3 구조) ============
O2_diff_flux = DO2 * grad(C)
O2_conv_flux = gamma_O2 * u * C

FO2 = (
      dot(O2_diff_flux, grad(vC)) * dx
    - dot(O2_conv_flux, grad(vC)) * dx
    + gamma_O2 * C * u_n_out * vC * ds_measure(16)
    + (I / (4 * F)) * MO2 * vC * ds_measure(13)                     # CL 산소 소모 (sink, 부호 +)
)

FC = FH2O + FO2

# ---- 최종 residual: NS/Darcy + 종보존 ----
Fres = Fu + FC


if MPI.COMM_WORLD.rank==0:
     print('problem start')
     
Problem=dolfinx.fem.petsc.NonlinearProblem(
    Fres,w,bcs=bc_all,
    petsc_options_prefix=f"Allof",
    form_compiler_options={"quadrature_degree": 4}, 
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": 1e-5,
        "snes_atol":1e-5,
        "snes_max_it": 50,
        "snes_linesearch_type": "bt",
        "snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)

if MPI.COMM_WORLD.rank==0:
     print('problem after')


#def singularity_test(u,P,S,C):





eta_history,I_history=[],[]

def checkpoint():
    return w.x.array.copy()

def restore(ckpt):
    w.x.array[:] = ckpt
    w.x.scatter_forward()

d_eta = 0.01
max_bisect = 4
eta_prev_val = float(eta.value) - 1e-6

eta_history, I_history,s_max_history = [], [],[]

s_expr_compiled = fem.Expression(s, V_C.element.interpolation_points)  
Iform = form((I/L) * ds_measure(13))


while float(eta.value) < 0.3:
    eta_target = float(eta.value) + d_eta
    pending = [eta_target]
    n_bisect = 0

    while pending:
        eta_now = pending[-1]
        ckpt = checkpoint()
        eta.value = eta_now

        Problem.solve()
        reason = Problem.solver.getConvergedReason()
        iters  = Problem.solver.getIterationNumber()
        converged = reason > 0

        if MPI.COMM_WORLD.rank == 0:
            print(f"eta={eta_now:.4f}: converged reason={reason}, iterations={iters}")

        if converged or n_bisect >= max_bisect:
            pending.pop()
            eta_prev_val = eta_now
            if not converged and MPI.COMM_WORLD.rank == 0:
                print(f"eta={eta_now:.4f}: bisection 한도 도달, 실패한 채로 넘어감")
        else:
            restore(ckpt)
            n_bisect += 1
            eta.value = eta_prev_val
            pending.append(0.5 * (eta_prev_val + eta_now))
            if MPI.COMM_WORLD.rank == 0:
                print(f"  실패 -> eta={pending[-1]:.4f}로 bisection 재시도 ({n_bisect}/{max_bisect})")

    eta.value = eta_target

    # --- 여기서부터는 원래 루프 안에 있던 내용, 그대로 이어붙이기 ---
    u_sol = w.sub(0).collapse(); u_sol.name = 'velocity'
    P_sol = w.sub(1).collapse(); P_sol.name = 'Pressure'
    C_sol = w.sub(2).collapse(); C_sol.name = 'Oxy Concentration'
    CW_sol = w.sub(3).collapse(); CW_sol.name = 'Water Concentration'

 
    s_proj = Function(V_C)
    s_proj.interpolate(s_expr_compiled)

    I_avg_a = fem.assemble_scalar(Iform)
    I_avg = domain.comm.allreduce(I_avg_a, op=MPI.SUM)

    if MPI.COMM_WORLD.rank == 0:
        print(f"  s range: [{s_proj.x.array.min():.4f}, {s_proj.x.array.max():.4f}]")
        print(f"  C range: [{C_sol.x.array.min():.4f}, {C_sol.x.array.max():.4f}]")
        print(f"  CW range: [{CW_sol.x.array.min():.4f}, {CW_sol.x.array.max():.4f}]")
        print(f' Average I  ,{I_avg}')

    eta_history.append(float(eta.value))
    I_history.append(float(I_avg))
    s_max_history.append(float(s_proj.x.array.max()))



end=time.time()

if MPI.COMM_WORLD.rank==0:
    print(f'runtime{(end-start)/60} minute')

    I_history_np=np.array(I_history)*1e-4

    plt.figure()
    plt.scatter(I_history_np,eta_history)
    plt.plot(I_history_np,eta_history)
    plt.ylabel('eta')
    plt.xlabel('I_avg A/cm2')
    plt.title('eta and I_avg')
    plt.show()

    plt.figure()
    plt.scatter(I_history_np,s_max_history)
    plt.plot(I_history_np,s_max_history)
    plt.ylabel('smax')
    plt.xlabel('I_avg A/cm2')
    plt.title('Max s and I_avg')
    plt.show()


V1 = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
u_viz = Function(V1)
u_viz.interpolate(u_sol)   # u_sol = w.sub(0).collapse() (이미 만들어두신 것)
u_viz.name='u_GDL'

with VTXWriter(domain.comm, "/home/ss/ChemFrontier/velo_final.bp", [u_viz,darcy_switch]) as vtx:
    vtx.write(0.0)

with VTXWriter(domain.comm, "/home/ss/ChemFrontier/Concen_final.bp", [C_sol]) as vtx:
    vtx.write(0.0)

with VTXWriter(domain.comm, "/home/ss/ChemFrontier/Satu_final.bp", [s_proj]) as vtx:
    vtx.write(0.0)

'''Info    : 15 entities
Info    : 110107 nodes
Info    : 223546 elements  '''

'''Info    : 15 entities
Info    : 5972 nodes
Info    : 12609 elements'''

'''Info    : 15 entities
Info    : 5972 nodes
Info    : 12609 elements'''

'''Info    : 15 entities
Info    : 507 nodes
Info    : 1112 elements
'''

'''Info    : 15 entities
Info    : 3897 nodes
Info    : 8292 elements
'''