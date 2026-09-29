# ブジネスク解：球座標から導出したフォン・ミーゼス応力の可視化 (nu=0.5専用)

import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [N] (1.0 nN)
G = 10*1.0e6  # せん断弾性率 [Pa] (10 MPa)
nu = 0.5      # ポアソン比 (この公式は 0.5専用)

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(-30, 30, 300)
z_vec = np.linspace(-20, 40, 300)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 応力の計算用配列の初期化 (初期値は NaN)
u_mises_MPa = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域 (z >= 0) のみに数式を適用
mask_elastic = (Z_mesh >= 0)
R_dist = np.sqrt(R_mesh**2 + Z_mesh**2)
mask_singularity = (R_dist < 0.1)  # 特異点回避
valid_idx = mask_elastic & ~mask_singularity

# 単位換算（m単位へ）
R_SI = R_dist[valid_idx] * 1e-9
Z_SI = Z_mesh[valid_idx] * 1e-9

# 【球座標から導出したフォン・ミーゼス応力】x = cos(theta) = z / R を用いた計算
x = Z_SI / R_SI
# 公式: mises = (3*P / (2*pi*R^2)) * x
mises_formula_Pa = (3.0 * P / (2.0 * np.pi * R_SI**2)) * x

# 計算結果を MPa 単位に換算して格納
u_mises_MPa[valid_idx] = mises_formula_Pa / 1.0e6

# 5. 可視化
fig, ax = plt.subplots(figsize=(8, 6))

# カラーマップの上限を設定 (MPa単位)
vmax_val = 3.0  
levels_setting = np.linspace(0, vmax_val, 51)

contour = ax.contourf(R_mesh, Z_mesh, u_mises_MPa, levels=levels_setting, cmap='viridis', extend='max', zorder=1)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('von Mises Stress, $\sigma_{Mises}$ / MPa', fontsize=12)

# 弾性体との境界を線で表現
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)

# 荷重点を矢印で表現
ax.annotate('', xy=(0, 0), xytext=(0, -10), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=4)
ax.text(0, -13, '$P$', color='red', ha='center', va='center', fontsize=14, zorder=4)

# パラメータ書き込み
param_text = f"Parameters:\n $P$ = {P*1e9:.1f} nN\n $G$ = {G/1e6:.1f} MPa\n $\\nu$ = {nu:.1f}"
props = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(15, -10, param_text, transform=ax.transData, fontsize=11, bbox=props, zorder=5)

# 軸の設定
ax.set_title('Boussinesq Solution (von Mises Stress / Spherical Formula)', fontsize=14, pad=15)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 40)
ax.invert_yaxis()  # 下向き正
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

fig.savefig('./png/Boussinesq_Mises_spherical.png', dpi=300)

plt.tight_layout()
plt.show()
