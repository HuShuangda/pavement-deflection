"""
无限大弹性层状体系沉降计算
==========================
基于 Burmister 多层弹性理论与状态空间传递矩阵法。

━━━ 输入 / 输出单位约定 ━━━
  层厚 hi        : 厘米  (cm)
  弹性模量 Ei    : 兆帕  (MPa)
  泊松比 νi      : 无量纲
  均布合力 P     : 千牛  (kN)    ← 总轴力，均布于荷载圆面积
  荷载圆半径 r   : 厘米  (cm)
  ─────────────────────────
  输出沉降 w     : 0.01 毫米      ← 即"百分之一毫米"

━━━ 模型 ━━━
  - N 层弹性体，第 1..N-1 层厚度有限 (hi, Ei, νi)
  - 第 N 层（最底层）厚度无限大 (E0, ν0)
  - 表面作用均布圆形荷载：合力 P，半径 r
  - 求圆心处表面沉降 w(0, 0)

━━━ 内部计算单位 ━━━
  内部统一采用 N–mm–MPa 制（1 MPa = 1 N/mm²），保证量纲自洽。
  接口层完成 cm↔mm、kN↔N、mm↔0.01mm 的换算。

━━━ 理论 ━━━
  轴对称问题 → Hankel 变换 → 关于 z 的 ODE 系统
  状态向量 X = [ū, w̄, σ̄_z, τ̄_rz]ᵀ 满足 dX/dz = A·X
  传递矩阵 T_i = expm(A_i · h_i)
  半无限底层：衰减子空间约束（从 A 矩阵广义特征向量解析导出）
  表面边界：σ̄_z(ξ,0) = -p·a·J₁(ξa)/ξ,  τ̄_rz(ξ,0) = 0
  逆变换：w(0,0) = ∫₀^∞ w̄(ξ,0)·ξ dξ
  其中 p = P/(πa²) 为均布荷载强度，a 为荷载半径（内部均用 mm）。
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.special import j1

# ──────────────────────────────────────────────
# 单位换算常量
# ──────────────────────────────────────────────

CM_TO_MM = 10.0          # 厘米 → 毫米
KN_TO_N  = 1000.0        # 千牛 → 牛
MM_TO_001MM = 100.0      # 毫米 → 0.01毫米

# ──────────────────────────────────────────────
# 核心求解器
# ──────────────────────────────────────────────

class LayeredElasticSystem:
    """
    多层弹性体系（底层半无限），圆心沉降计算。

    输入单位：厚度 cm，模量 MPa，合力 kN，半径 cm。
    输出单位：沉降 0.01 mm。

    Parameters
    ----------
    layers : list of tuple
        [(h1_cm, E1_MPa, ν1), (h2_cm, E2_MPa, ν2), ..., (inf, E0_MPa, ν0)]
        厚度单位 cm；模量单位 MPa；最后一层厚度填 float('inf')。
    force_kN : float
        均布荷载合力 P（kN）。
    radius_cm : float
        荷载圆半径 r（cm）。
    """

    def __init__(self, layers, force_kN, radius_cm):
        self.layers = layers
        self.N = len(layers)
        self.force_kN = force_kN
        self.radius_cm = radius_cm
        self._validate()

        # ── 内部统一单位：N, mm, MPa ──
        self._a_mm = radius_cm * CM_TO_MM                    # 半径 → mm
        self._P_N  = force_kN * KN_TO_N                      # 合力 → N
        area = np.pi * self._a_mm**2                         # 荷载圆面积 mm²
        self._p_MPa = self._P_N / area                       # 均布强度 MPa (= N/mm²)

        # 内部层参数：厚度转 mm，模量保持 MPa
        self._layers_mm = []
        for h_cm, E, nu in layers:
            h_mm = float('inf') if np.isinf(h_cm) else h_cm * CM_TO_MM
            self._layers_mm.append((h_mm, E, nu))

    def _validate(self):
        for i, (h, E, nu) in enumerate(self.layers):
            if E <= 0:
                raise ValueError(f"第 {i+1} 层弹性模量必须 > 0（当前 {E} MPa）")
            if not (0 < nu < 0.5):
                raise ValueError(f"第 {i+1} 层泊松比必须在 (0, 0.5) 范围内（当前 {nu}）")
        if self.N < 1:
            raise ValueError("至少需要 1 层")
        if self.radius_cm <= 0:
            raise ValueError("荷载圆半径必须 > 0")
        if self.force_kN <= 0:
            raise ValueError("合力必须 > 0")
        # 除最后一层外，厚度必须有限
        for i in range(self.N - 1):
            if np.isinf(self.layers[i][0]):
                raise ValueError(f"第 {i+1} 层（非底层）厚度不能为无限大")

    # ── 以下为内部计算方法，全部使用 N–mm–MPa 制 ──

    @staticmethod
    def _A_matrix(xi, E, nu):
        """状态空间矩阵 A (4×4)，dX/dz = A·X。"""
        if nu == 0.5:
            nu = 0.499999
        G = E / (2 * (1 + nu))
        A = np.zeros((4, 4))
        A[0, 1] = xi
        A[0, 3] = 1.0 / G
        A[1, 0] = -nu * xi / (1 - nu)
        A[1, 2] = (1 - 2 * nu) / (2 * G * (1 - nu))
        A[2, 3] = -xi
        A[3, 0] = 2 * G * xi**2 / (1 - nu)
        A[3, 2] = nu * xi / (1 - nu)
        return A

    @staticmethod
    def _decay_constraint(xi, E, nu):
        """
        半无限底层衰减子空间约束（底层顶面 z=0 处）。

        从 A 矩阵广义特征向量解析导出（特征值 -ξ，2×2 Jordan 块）：

          (I)  2Gξ·ū  − (1−2ν)·σ̄ + 2(1−ν)·τ̄ = 0
          (II) 2Gξ·w̄  + 2(1−ν)·σ̄ − (1−2ν)·τ̄ = 0

        Returns: R (2×4)，满足 R @ X = 0。
        """
        if nu == 0.5:
            nu = 0.499999
        G = E / (2 * (1 + nu))
        a1 = 1 - 2 * nu
        a2 = 2 * (1 - nu)
        R = np.zeros((2, 4))
        R[0, 0] = 2 * G * xi
        R[0, 2] = -a1
        R[0, 3] = a2
        R[1, 1] = 2 * G * xi
        R[1, 2] = a2
        R[1, 3] = -a1
        return R

    def _solve_at_xi(self, xi):
        """对给定 ξ，返回表面 w̄(ξ, 0)（内部 mm 单位）。"""
        p = self._p_MPa
        a = self._a_mm

        # 表面应力 Hankel 变换
        if xi < 1e-14:
            sigma_bar = -p * a**2 / 2.0
        else:
            sigma_bar = -p * a * j1(xi * a) / xi
        tau_bar = 0.0

        # 单层半无限体
        if self.N == 1:
            E, nu = self._layers_mm[0][1], self._layers_mm[0][2]
            R = self._decay_constraint(xi, E, nu)
            A_sys = np.array([[R[0, 0], R[0, 1]],
                              [R[1, 0], R[1, 1]]])
            b_sys = np.array([-(R[0, 2] * sigma_bar + R[0, 3] * tau_bar),
                              -(R[1, 2] * sigma_bar + R[1, 3] * tau_bar)])
            sol = np.linalg.solve(A_sys, b_sys)
            return sol[1]

        # 多层传递矩阵法
        T_total = np.eye(4)
        for i in range(self.N - 1):
            h, E, nu = self._layers_mm[i]
            A_i = self._A_matrix(xi, E, nu)
            T_i = expm(A_i * h)
            T_total = T_i @ T_total

        E0, nu0 = self._layers_mm[-1][1], self._layers_mm[-1][2]
        R = self._decay_constraint(xi, E0, nu0)
        RT = R @ T_total

        A_sys = np.array([[RT[0, 0], RT[0, 1]],
                          [RT[1, 0], RT[1, 1]]])
        b_sys = np.array([-(RT[0, 2] * sigma_bar + RT[0, 3] * tau_bar),
                          -(RT[1, 2] * sigma_bar + RT[1, 3] * tau_bar)])
        sol = np.linalg.solve(A_sys, b_sys)
        return sol[1]

    def _surface_deflection_mm(self, xi_max=None):
        """圆心沉降，内部 mm 单位。w = ∫₀^∞ w̄·ξ dξ"""
        a = self._a_mm
        if xi_max is None:
            xi_max = 40.0 / a

        def integrand(xi):
                return self._solve_at_xi(xi) * xi


        result, _ = quad(integrand, 0, xi_max,
                         limit=200, points=[1.0/a, 5.0/a, 10.0/a],
                         epsabs=1e-12, epsrel=1e-10)
        return result

    # ── 公开接口 ──
    def center_settlement(self, xi_max=None):
        """
        计算圆心处表面沉降。

        Returns
        -------
        float
            沉降值，单位 **0.01 mm**（百分之一毫米）。
        """
        w_mm = self._surface_deflection_mm(xi_max)
        return w_mm * MM_TO_001MM

    def center_settlement_mm(self, xi_max=None):
        """沉降值，单位 mm（便于对照）。"""
        return self._surface_deflection_mm(xi_max)


# ──────────────────────────────────────────────
# 使用示例
# ──────────────────────────────────────────────

if __name__ == '__main__':
    #   面层(cm, MPa, ν) | 基层(cm, MPa, ν) | 路基(∞, MPa, ν)
    layers = [
        (4, 9800, 0.25),
        (6, 12500, 0.25),
        (8, 11500, 0.25),
        (12, 9000, 0.25),
        (18, 14000, 0.25),
        (18, 12000, 0.25),
        (15, 300, 0.35),
        (float('inf'), 60, 0.4),  # 路基 半无限
    ]
    P_kN, r_cm = 50, 15            # 合力 50 kN，半径 15 cm

    les = LayeredElasticSystem(layers, force_kN=P_kN, radius_cm=r_cm)
    w = les.center_settlement()

    print(f"圆心处沉降 w = {w:.2f} (0.01 mm) = {w/100:.4f} mm")
