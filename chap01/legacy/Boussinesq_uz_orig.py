# ブジネスク解（変位u_z)

import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9    # 荷重 [nN]
G = 10*1.0e6  # せん断弾性率 [Pa]
nu = 0.4      # ポアソン比

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(-30, 30, 300)
z_vec = np.linspace(-20, 40, 300)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 変位 u_z の計算用配列の初期化 (初期値は NaN)
u_z_nm = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域 (z >= 0) のみにブジネスク解を適用
mask_elastic = (Z_mesh >= 0)
R_dist = np.sqrt(R_mesh**2 + Z_mesh**2)
mask_singularity = (R_dist < 0.1)

# 計算対象のインデックス (弾性体内部かつ特異点ではない場所)
valid_idx = mask_elastic & ~mask_singularity

# 単位換算をして計算 (m単位へ)
R_SI = R_dist[valid_idx] * 1e-9
Z_SI = Z_mesh[valid_idx] * 1e-9

# u_z の計算と nm への換算
u_z_SI = (P / (4 * np.pi * G * R_SI)) * ((Z_SI**2 / R_SI**2) + 2 * (1 - nu))
u_z_nm[valid_idx] = u_z_SI * 1e9

# 5. 可視化
fig, ax = plt.subplots(figsize=(8, 6))

# カラースケールの上限（vmax）を設定
vmax_val = 5.0  # ここにお好みの最大値（nm単位）を入力してください
levels_setting = np.linspace(0, vmax_val, 51)  # 0からvmax_valまでを50等分（51個の境界値）

# 等高線プロット
contour = ax.contourf(R_mesh, Z_mesh, u_z_nm, levels=levels_setting, cmap='viridis', extend='max', zorder=1)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Vertical Displacement, $u_z$ /nm', fontsize=12)

# 弾性体との境界を線で表現
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)    # 表面の境界線

# 荷重点を矢印で表現 (中心 r=0, z=0 に向かって下向き)
ax.annotate('', xy=(0, 0), xytext=(0, -10),
            arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=4)
ax.text(0, -13, '$P$', color='red', ha='center', va='center', fontsize=14, zorder=4)

# 領域のテキストラベル
ax.text(-20, -10, 'Free Space', color='gray', fontsize=11, ha='center', va='center', zorder=3)
ax.text(-20, 25, 'Elastic Body', color='white', fontsize=11, ha='center', va='center', zorder=3)

# 【追加機能】パラメータ (P, G, nu) を画面内のテキストボックスとして書き込み
param_text = (
    f"Parameters:\n"
    f" $P$ = {P*1e9:.1f} nN\n"
    f" $G$ = {G/1e6:.1f} MPa\n"
    f" $\\nu$ = {nu:.1f}"
)
props = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(15, -10, param_text, transform=ax.transData, fontsize=11,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定
ax.set_title('Boussinesq Solution (displacement, $u_z$)', fontsize=14, pad=15)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 40)
ax.invert_yaxis()  # 下向き正
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

fig.savefig('./png/Boussinesq_uz.png', dpi=300)

plt.tight_layout()
plt.show()
