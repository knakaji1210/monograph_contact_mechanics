# ブジネスク解：中心軸上（r=0）におけるフォン・ミーゼス応力の理論式と力技の比較検証

import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [N] (1.0 nN)
G = 10*1.0e6  # せん断弾性率 [Pa] (10 MPa)
nu = 0.4      # ポアソン比 (0.5など自由に変更してテスト可能)

# 2. 中心軸上 (r=0) の深さ方向のグリッドを設定 (単位: nm)
# 特異点 (z=0) を回避するため、深さ 1.0 nm から 40 nm までを設定
z_vec = np.linspace(1.0, 40, 200)
r_val = 0.0  # 中心軸上

# 単位換算 (m単位の配列へ)
Z_SI = z_vec * 1e-9
r_SI = r_val * 1e-9
R_SI = np.sqrt(r_SI**2 + Z_SI**2)  # 中心軸上なので R_SI == Z_SI となります

# ==========================================
# アプローチ1: 4成分からの「力技」による計算
# ==========================================
sigma_z = -(3.0 * P / (2.0 * np.pi)) * (Z_SI**3 / R_SI**5)
tau_zr  = -(3.0 * P / (2.0 * np.pi)) * (r_SI * Z_SI**2 / R_SI**5)  # r=0 なので 0

# 各成分の極限（r->0）における正しい軸上値を直接計算
# 軸上では分母の (1 + z/R) は (1 + 1) = 2 になります
sigma_theta = ((1.0 - 2.0 * nu) * P / (2.0 * np.pi * R_SI**2)) * ((Z_SI / R_SI) - (1.0 / (1.0 + Z_SI / R_SI)))
sigma_r = (P / (2.0 * np.pi * R_SI**2)) * (-(3.0 * r_SI**2 * Z_SI / R_SI**3) + (1.0 - 2.0 * nu) * (1.0 / (1.0 + Z_SI / R_SI)))

# ミーゼス応力の定義式に力技で代入
mises_force_Pa = np.sqrt(
    0.5 * ((sigma_r - sigma_z)**2 + (sigma_z - sigma_theta)**2 + (sigma_theta - sigma_r)**2) + 3.0 * tau_zr**2
)
mises_force_MPa = mises_force_Pa / 1.0e6  # MPaに換算

# ==========================================
# アプローチ2: 中心軸上の厳密解
# 公式: sigma_Mises = (P / (2*pi*z^2)) * (7/2 - nu)
# ==========================================
mises_exact_Pa = (P / (2.0 * np.pi * Z_SI**2)) * (3.5 - nu)
mises_exact_MPa = mises_exact_Pa / 1.0e6  # MPaに換算

# ==========================================
# 数値的同一性の検証（最大絶対誤差の評価）
# ==========================================
max_diff = np.max(np.abs(mises_force_Pa - mises_exact_Pa))
print(f"【検証結果】中心軸上（r=0）における最大絶対誤差: {max_diff:.6e} Pa")

# 3. 可視化（2つの結果をグラフにプロットして比較）
fig, ax = plt.subplots(figsize=(7, 5))

# 2つの曲線を重ねて描画（完全に重なるため、一方は破線に）
ax.plot(z_vec, mises_force_MPa, label='Direct Component Calculation', color='blue', linewidth=2.5)
ax.plot(z_vec, mises_exact_MPa, label='Exact Analytical Solution', color='red', linestyle='--', linewidth=2.5)

# グラフの装飾
ax.set_title('von Mises Stress on Central Axis ($r=0$)', fontsize=13, pad=15)
ax.set_xlabel('Depth, $z$ / nm', fontsize=12)
ax.set_ylabel('von Mises Stress, $\sigma_{Mises}$ / MPa', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=11)

# パラメータ表示ボックス
param_text = f"Parameters:\n $P$ = {P*1e9:.1f} nN\n $G$ = {G/1e6:.1f} MPa\n $\\nu$ = {nu:.1f}"
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(0.65, 0.65, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

fig.savefig('./png/Boussinesq_Mises_comp_axis.png', dpi=300)

plt.tight_layout()
plt.show()
