# ブジネスク解：フォン・ミーゼス応力の全領域誤差検証

import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [N] (1.0 nN)
G = 10*1.0e6  # せん断弾性率 [Pa] (10 MPa)
nu = 0.5      # ポアソン比 (非圧縮性 0.5 固定)

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(-30, 30, 300)
z_vec = np.linspace(-20, 40, 300)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 誤差計算用配列の初期化 (初期値は NaN)
error_Pa = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域 (z >= 0) のみマスク処理
mask_elastic = (Z_mesh >= 0)
R_dist = np.sqrt(R_mesh**2 + Z_mesh**2)

# 荷重点直下の数値的な桁落ちを防ぐため、原点から2 nm以内の領域を除外
mask_near_singularity = (R_dist < 2.0)
valid_idx_clean = mask_elastic & ~mask_near_singularity

# 単位換算（m単位へ）
R_SI = R_dist[valid_idx_clean] * 1e-9
Z_SI = Z_mesh[valid_idx_clean] * 1e-9
r_SI = R_mesh[valid_idx_clean] * 1e-9

# ==========================================
# アプローチ1: 応力4成分からの「力技」計算
# ==========================================
sigma_z = -(3.0 * P / (2.0 * np.pi)) * (Z_SI**3 / R_SI**5)
tau_zr  = -(3.0 * P / (2.0 * np.pi)) * (r_SI * Z_SI**2 / R_SI**5)

# nu = 0.5 のため、以下の項は自動的に 0 になる
sigma_theta = ((1.0 - 2.0 * nu) * P / (2.0 * np.pi * R_SI**2)) * ((Z_SI / R_SI) - (1.0 / (1.0 + Z_SI / R_SI)))
sigma_r = (P / (2.0 * np.pi * R_SI**2)) * (-(3.0 * r_SI**2 * Z_SI / R_SI**3) + (1.0 - 2.0 * nu) * (1.0 / (1.0 + Z_SI / R_SI)))

mises_force = np.sqrt(
    0.5 * ((sigma_r - sigma_z)**2 + (sigma_z - sigma_theta)**2 + (sigma_theta - sigma_r)**2) + 3.0 * tau_zr**2
)

# ==========================================
# アプローチ2: 【球座標から導出したフォン・ミーゼス応力】
# 公式: sigma_Mises = (3 * P / (2 * pi * R^2)) * x
# ==========================================
x = Z_SI / R_SI  # cos(theta)
mises_formula = (3.0 * P / (2.0 * np.pi * R_SI**2)) * x

# ==========================================
# 絶対誤差の計算
# ==========================================
diff_array_clean = np.abs(mises_force - mises_formula)
max_diff_clean = np.max(diff_array_clean)

# 誤差を配列に格納
error_Pa[valid_idx_clean] = diff_array_clean

# 5. 可視化 (誤差の分布プロット)
fig, ax = plt.subplots(figsize=(8, 6))

# 誤差が完全にゼロであることを示すため、カラー上限は極小スケール (0 〜 1e-5 Pa) で固定
vmax_val = 1.0e-5  
levels_setting = np.linspace(0, vmax_val, 51)

contour = ax.contourf(R_mesh, Z_mesh, error_Pa, levels=levels_setting, cmap='coolwarm', extend='max', zorder=1)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Absolute Error, |$\sigma_{force}$ - $\sigma_{formula}$| / Pa', fontsize=12)

# 弾性体との境界を線で表現
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)

# 荷重点を矢印で表現
ax.annotate('', xy=(0, 0), xytext=(0, -10), arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=4)
ax.text(0, -13, '$P$', color='red', ha='center', va='center', fontsize=14, zorder=4)

# 画面内のテキストボックスにパラメータと最大絶対誤差 (Max Error) を明記
param_text = (
    f"Parameters:\n"
    f" $P$ = {P*1e9:.1f} nN\n"
    f" $G$ = {G/1e6:.1f} MPa\n"
    f" $\\nu$ = {nu:.1f}\n\n"
    f"Max Error (Outside 2nm):\n"
    f" {max_diff_clean:.2e} Pa"
)
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(10, -10, param_text, transform=ax.transData, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定
ax.set_title('Absolute Error Map (Analytical Formula ($\\nu$ = 0.5) vs Direct Components)', fontsize=13, pad=15)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 40)
ax.invert_yaxis()  # 下向き正
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

fig.savefig('./png/Boussinesq_Mises_comp_area.png', dpi=300)

plt.tight_layout()
plt.show()
