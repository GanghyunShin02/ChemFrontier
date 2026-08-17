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


meshdata=gmshio.read_from_msh('/home/ss/ChemFrontier/Cathode3.msh',MPI.COMM_WORLD,0,2)
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

sigma_mem = Constant(domain, default_scalar_type(8.0))   # 막 이온전도도, S/m (수화된 Nafion 기준)

alpc     = Constant(domain, default_scalar_type(0.5))
epsilon  = Constant(domain,default_scalar_type(0.5))

dH = Constant(domain, default_scalar_type(3e-4))
H1 = Constant(domain, default_scalar_type(1e-3))
L  = Constant(domain, default_scalar_type(0.05))

nuAir=Constant(domain,default_scalar_type(2.1e-5))
muAir=Constant(domain,default_scalar_type(2.09e-5))


tau=Constant(domain,default_scalar_type(1.5))

I_ref = Constant(domain, default_scalar_type(100))   # A/m^2, 고정 전류밀도
eta=Constant(domain,default_scalar_type(0.05)) # 과전압 시작값.
V_oc=Constant(domain,default_scalar_type(1.2)) # 표준환원전위

Pv_sat = Constant(domain, default_scalar_type(47400))   # Pa, 353K 물 포화증기압
Mair   = Constant(domain, default_scalar_type(0.029))   # kg/mol



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


rhoAir = Pc*Mair/(R*T)  
rhoO2=Pc*MO2/(R*T)

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

DO2=((epsilon*(1-s))**tau)*DgO2


# ---- C_H2O를 s의 대수함수로 계산 (식 6, 21, 22) ----
rho_mix = rhowater*s + rhoAir*(1-s)             # 식(5)



Fu=(
      dot(dot(u, nabla_grad(u)), v) * dx        # 대류항
    + nuAir * inner(grad(u), grad(v)) * dx      # 점성항
    - p * div(v) * dx                           # 압력-속도 커플링
    + q * div(u) * dx                           # 연속방정식 (필수, 이전에 빠졌던 부분)
    + darcy_switch * (nuAir/K) * dot(u, v) * dx # GDL Darcy 저항 소스텀
)



n = ufl.FacetNormal(domain)


# ---- Tafel 전류 (논문 식 3.5) ----
# alpc를 써야 함 (alp=0.5는 anodic 쪽), CO2_in Function 대신 상수 기준농도 사용
CO2_ref = Constant(domain, default_scalar_type(0.21))

C_smooth_eps = 1e-8   # CO2_ref=0.21 대비 작은 값, 필요시 조정
C_pos = 0.5*(C + ufl.sqrt(C**2 + C_smooth_eps**2))


I = (1 - s) * I_ref * (C_pos / CO2_ref) * ufl.exp(alpc * F * eta / (R * T))

# ---- 아웃렛 클리핑 (논문 §5.2.1 핵심 stabilization) ----
u_n = dot(u, n)
u_n_out = ufl.conditional(ufl.gt(u_n, 0), u_n, 0)   # max(u·n, 0), 역류 시 0으로 클리핑

# ============ 물/포화도 수송 (식 3.2 구조, s가 1차 미지수) ============
#water_diff_flux = D_cap * grad(s)
water_conv_flux = gamma_H2O * rho_mix * u *CW

DgH2O = Constant(domain, default_scalar_type(2.56e-5))   # 수증기-공기 확산계수, m^2/s

dCw = 1e-6
ds_dCw = (s_expr(CW + dCw) - s_expr(CW - dCw)) / (2*dCw)

D_eff_w = epsilon*(1 - s)*DgH2O + D_cap*ds_dCw
water_diff_flux = rho_mix * D_eff_w * grad(CW)

FH2O = (
      dot(water_diff_flux, grad(vW)) * dx
    - dot(water_conv_flux, grad(vW)) * dx
    + gamma_H2O * rho_mix  * u_n_out * vW *CW* ds_measure(16)
    - (I / (2 * F)) * MH2O * vW * ds_measure(13)
)
# ============ 산소 수송 (식 3.3 구조) ============
O2_diff_flux = rhoO2*DO2 * grad(C)
O2_conv_flux = rhoO2*gamma_O2 * u * C

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
        "snes_rtol": 1e-8,
        "snes_atol":1e-8,
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

s_expr_compiled = fem.Expression(s, V_C.element.interpolation_points)
Iform = form((I/L) * ds_measure(13))


def checkpoint():
    return w.x.array.copy()


def restore(ckpt):
    w.x.array[:] = ckpt
    w.x.scatter_forward()


def reset_state(eta_start=0.05):
    w.x.array[:] = 0.0
    w.sub(0).interpolate(lambda x: np.vstack((np.full(x.shape[1], 0.4), np.zeros(x.shape[1]))),
                          cells0=channeltag)
    w.sub(2).interpolate(lambda x: np.full(x.shape[1], 0.21))
    w.sub(3).interpolate(lambda x: np.full(x.shape[1], 0.01))
    w.x.scatter_forward()
    eta.value = eta_start


def run_eta_sweep(eta_max=0.8, d_eta_init=0.01, max_bisect=4):
    d_eta = d_eta_init
    eta_prev_val = float(eta.value) - 1e-6
    hist = {"eta": [], "I": [], "s_max": [], "V": [],"C_min":[]}

    while float(eta.value) < eta_max:
        eta_target = float(eta.value) + d_eta
        pending = [eta_target]
        n_bisect = 0

        while pending:
            eta_now = pending[-1]
            ckpt = checkpoint()
            eta.value = eta_now

            Problem.solve()
            converged = Problem.solver.getConvergedReason() > 0

            if converged or n_bisect >= max_bisect:
                pending.pop()
                eta_prev_val = eta_now
            else:
                restore(ckpt)
                n_bisect += 1
                eta.value = eta_prev_val
                pending.append(0.5 * (eta_prev_val + eta_now))

        eta.value = eta_target

        s_proj = Function(V_C)
        s_proj.interpolate(s_expr_compiled)


        C_sol=w.sub(2).collapse()

        I_avg = domain.comm.allreduce(fem.assemble_scalar(Iform), op=MPI.SUM)
        V_now = V_oc.value - float(eta.value) - (I_avg * float(dH.value)) / sigma_mem.value

        if MPI.COMM_WORLD.rank == 0:

            coords = V_C.tabulate_dof_coordinates()

            idx_min = np.argmin(C_sol.x.array)
            print("C_min 위치:", coords[idx_min], " 값:", C_sol.x.array[idx_min])

            C_at_CL_avg = fem.assemble_scalar(fem.form(C * ds_measure(13)))
            CL_length = fem.assemble_scalar(fem.form(1 * ds_measure(13)))
            print("CL 평균 C:", C_at_CL_avg / CL_length)

            C_at_CL_min = fem.assemble_scalar(fem.form(ufl.conditional(ufl.lt(C,1e10), C, 1e10) * ds_measure(13)))
            # 또는 DOF 좌표로 직접 필터링
            coords = V_C.tabulate_dof_coordinates()
            C_sol = w.sub(2).collapse()
            cl_mask = np.isclose(coords[:,1], 0.0012, atol=1e-6)
            print("CL 위 C 값들 (min/max/mean):", 
                C_sol.x.array[cl_mask].min(), 
                C_sol.x.array[cl_mask].max(), 
                C_sol.x.array[cl_mask].mean())



            print(f"  eta={float(eta.value):.4f}  I={I_avg:.3f}  smax={s_proj.x.array.max():.4f}  V={V_now:.3f} C_min={C_sol.x.array.min()}" , flush=True)

        hist["eta"].append(float(eta.value))
        hist["I"].append(I_avg)
        hist["s_max"].append(float(s_proj.x.array.max()))
        hist["V"].append(V_now)
        hist["C_min"].append(float(C_sol.x.array.min()))

        if V_now < 0:
            break

    return hist


CL_avg_form = form(C * ds_measure(13))
CL_length_form = form(1 * ds_measure(13))
CL_len = domain.comm.allreduce(fem.assemble_scalar(CL_length_form), op=MPI.SUM)

epsilon.value = 0.5
tau.value = 1.1
reset_state(eta_start=-50)
Problem.solve()

CL_avg = domain.comm.allreduce(fem.assemble_scalar(CL_avg_form), op=MPI.SUM)
if MPI.COMM_WORLD.rank == 0:
    print("CL(y=0.0012) 평균 C:", CL_avg / CL_len, flush=True)

for eta_test in [1e-6, 0.001, 0.01, 0.02, 0.04, 0.06]:
    eta.value = eta_test
    Problem.solve()
    CL_avg = domain.comm.allreduce(fem.assemble_scalar(CL_avg_form), op=MPI.SUM)
    if MPI.COMM_WORLD.rank == 0:
        print(f"eta={eta_test}: CL 평균 C = {CL_avg/CL_len}", flush=True)


'''


# ================= epsilon 스윕 (tau 고정) =================
tau_fixed = 1.1
epsilon_vals = [ 0.7]



results_eps = {}
tau.value = tau_fixed
for eps_v in epsilon_vals:
    epsilon.value = eps_v
    reset_state(eta_start=0.05)
    if MPI.COMM_WORLD.rank == 0:
        print(f"[epsilon={eps_v}]", flush=True)
    results_eps[eps_v] = run_eta_sweep(eta_max=0.8)


end = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f'runtime {(end-start)/60:.2f} minute', flush=True)


    plt.figure()
    for eps_v, hist in results_eps.items():
        I_cm2 = np.array(hist["I"]) * 1e-4
        plt.plot(I_cm2, hist["eta"], marker='o', label=f"epsilon={eps_v}")
    plt.xlabel("I (A/cm$^2$)")
    plt.ylabel("eta (V)")
    plt.title(f"Effect of porosity (tau={tau_fixed})")
    plt.legend()
    plt.show()



    plt.figure()
    for eps_v, hist in results_eps.items():
        I_cm2 = np.array(hist["I"]) * 1e-4
        plt.plot(I_cm2, hist["s_max"], marker='o', label=f"epsilon={eps_v}")
    plt.xlabel("I (A/cm$^2$)")
    plt.ylabel("s_max")
    plt.title(f"Effect of porosity on s_max (tau={tau_fixed})")
    plt.legend()
    plt.show()

    plt.figure()
    for eps_v, hist in results_eps.items():
        I_cm2 = np.array(hist["I"]) * 1e-4
        plt.plot(I_cm2, hist["C_min"], marker='o', label=f"epsilon={eps_v}")
    plt.xlabel("I (A/cm$^2$)")
    plt.ylabel("Oxygen C_min")
    plt.title(f"Effect of porosity on Oxyge C_min (tau={tau_fixed})")
    plt.legend()
    plt.show()

    '''
