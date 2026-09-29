# ブジネスク解（円柱座標系 r >= 0 における変位ベクトルのノルム ||u||）

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [N] (1.0 nN)
G = 10*1.0e6  # せん断弾性率 [Pa] (10 MPa)
nu = 0.4      # ポアソン比

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(1e-5, 30, 300)
z_vec = np.linspace(0, 30, 300)  # 深さ 30 nm 基準

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 変位ベクトルのノルム ||u|| の計算用配列の初期化
u_norm_nm = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域にブジネスク解を適用
R_dist = np.sqrt(R_mesh**2 + Z_mesh**2)
mask_singularity = (R_dist < 0.1)
valid_idx = ~mask_singularity

# 単位換算をして計算 (m単位へ)
R_SI = R_dist[valid_idx] * 1e-9
Z_SI = Z_mesh[valid_idx] * 1e-9
r_SI = R_mesh[valid_idx] * 1e-9  

# 各変位成分の計算 (SI単位系)
u_r_SI = (P / (4 * np.pi * G * R_SI)) * (
    (r_SI * Z_SI / R_SI**2) - ((1 - 2 * nu) * r_SI / (R_SI + Z_SI))
)
u_z_SI = (P / (4 * np.pi * G * R_SI)) * (
    (Z_SI**2 / R_SI**2) + 2 * (1 - nu)
)

# 変位ベクトルの L2 ノルム ||u|| = sqrt(u_r^2 + u_z^2) を計算
u_norm_SI = np.sqrt(u_r_SI**2 + u_z_SI**2)

# nm 単位に換算して格納
u_norm_nm[valid_idx] = u_norm_SI * 1e9

# 5. 可視化
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

# 色彩レンジ設定
vmax_val = 5.0  
levels_setting = np.linspace(-vmax_val, vmax_val, 101)  

# turbo カラーマップを適用（常に正の値なので境界破線は不要）
contour = ax.contourf(R_mesh, Z_mesh, u_norm_nm, levels=levels_setting, cmap='turbo', extend='both', zorder=1)

# カラーバーのラベル設定
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('Displacement Norm, $\|\mathbf{u}\|$ /nm', fontsize=11)

# 弾性体表面（z=0）の境界線
ax.axhline(0, color='black', linewidth=0.5, linestyle='-', zorder=3)    

# 荷重点を矢印で表現
ax.annotate('', xy=(0, 0), xytext=(0, -8),
            arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=4)
ax.text(0, -11, '$P$', color='red', ha='center', va='center', fontsize=14, zorder=4)

# 領域 of テキストラベル
ax.text(3, -4, 'Free Space', color='gray', fontsize=11, ha='left', va='center', zorder=3)
ax.text(3, 15, 'Elastic Body', color='black', fontsize=11, ha='left', va='center', zorder=3)

# パラメータ (P, G, nu) を書き込み
param_text = (
    f"$P$ = {P*1e9:.1f} nN\n"
    f"$G$ = {G/1e6:.1f} MPa\n"
    f"$\\nu$ = {nu:.1f}"
)
props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(18, -6, param_text, transform=ax.transData, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定（\mathbf{u} で統一）
ax.set_title('Boussinesq Solution (Norm, $\|\mathbf{u}\|$)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)  
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 保存
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Boussinesq_u_norm.png', dpi=300, bbox_inches='tight')

plt.show()
