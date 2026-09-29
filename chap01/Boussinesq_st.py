# ブジネスク解（円柱座標系 r >= 0 における円周方向応力 σ_θ）

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [N] (1.0 nN)
G = 10*1.0e6  # せん断弾性率 [Pa] (10 MPa)
nu = 0.4      # ポアソン比

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(1e-5, 30, 300)
z_vec = np.linspace(0, 30, 300)  # テンプレート基準：深さ 30 nm

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 円周方向応力 sigma_theta の計算用配列の初期化
sigma_theta_MPa = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域にブジネスク解を適用
R_dist = np.sqrt(R_mesh**2 + Z_mesh**2)
mask_singularity = (R_dist < 0.1)
valid_idx = ~mask_singularity

# 単位換算をして計算 (m単位へ)
R_SI = R_dist[valid_idx] * 1e-9
Z_SI = Z_mesh[valid_idx] * 1e-9

# 円周方向応力 sigma_theta の計算 (SI単位系: Pa)
sigma_theta_SI = ((1 - 2 * nu) * P / (2 * np.pi * R_SI**2)) * (
    (Z_SI / R_SI) - (1.0 / (1.0 + Z_SI / R_SI))
)

# MPa 単位に換算して格納 (1 MPa = 1e6 Pa)
sigma_theta_MPa[valid_idx] = sigma_theta_SI / 1e6

# 5. 可視化
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

# 色彩レンジ設定
vmax_val = 2.0  
levels_setting = np.linspace(-vmax_val, vmax_val, 101)  

# テンプレート基準：'turbo' カラーマップを適用
contour = ax.contourf(R_mesh, Z_mesh, sigma_theta_MPa, levels=levels_setting, cmap='turbo', extend='both', zorder=1)

# 応力の正負の境界線（sigma_theta = 0）に黒い破線を1本重ね書き
# （領域内に 0.0 の等高線が存在する場合のみプロット）
ax.contour(R_mesh, Z_mesh, sigma_theta_MPa, levels=[0.0], colors='black', linewidths=0.5, linestyles='--', zorder=2)

# カラーバーを下側に配置
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('Circumferential Stress, $\sigma_\\theta$ /MPa', fontsize=11)

# 弾性体表面（z=0）の境界線
ax.axhline(0, color='black', linewidth=0.5, linestyle='-', zorder=3)    

# 荷重点を矢印で表現
ax.annotate('', xy=(0, 0), xytext=(0, -8),
            arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=4)
ax.text(0, -11, '$P$', color='red', ha='center', va='center', fontsize=14, zorder=4)

# 領域のテキストラベル
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

# 軸の設定
ax.set_title('Boussinesq Solution (Stress, $\sigma_\\theta$)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)  
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 保存
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Boussinesq_st.png', dpi=300, bbox_inches='tight')

plt.show()
