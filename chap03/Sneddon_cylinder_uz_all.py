# 平底円柱圧子接触解（厳密解のダイレクト計算版：垂直変位 u_z ＆ 厳密圧力分布）
# Galinパラメータ xi を用いた解析解の可視化（修正版）

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数および計算条件の設定 (ナノスケール)
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
a_probe = 10.0e-9   # 平底円柱圧子の半径 a: 10 nm
a_SI = a_probe      # 円柱の場合、接触半径 a は圧子半径で固定

# 2. 絶対的な等高線基準の計算（nu=0.5 の delta を絶対基準として固定）
nu_base = 0.5
E_star_base = E / (1.0 - nu_base**2)
delta_base = P / (2.0 * E_star_base * a_SI)

# 現在のポアソン比 nu における実際の弾性諸元
E_star = E / (1.0 - nu**2)
delta_SI = P / (2.0 * E_star * a_SI)
p_avg = P / (np.pi * a_SI**2)       # 平均接触圧力 P / (pi * a^2)
p0 = P / (2.0 * np.pi * a_SI**2)    # 中心(r=0)の接触圧力 p(0) = p_avg / 2

# 3. 計算領域の設定 (単位: nm)
a_nm = a_SI * 1e9
r_vec = np.linspace(1e-5, 30, 300)  # 円柱座標系 r >= 0
z_vec = np.linspace(0, 30, 300)     # 深さ 30 nm (弾性体内部のみ)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 各メッシュ点の物理量をm単位に換算 (ゼロ割り・特異点回避のため極小のオフセット 1e-20 を追加)
r_SI = R_mesh * 1e-9 + 1e-20
z_SI = Z_mesh * 1e-9 + 1e-20

# 4. 幾何学空間パラメータ xi (正の実根) の計算
# 解くべき方程式: xi^2 + (a^2 - r^2 - z^2)*xi - a^2*z^2 = 0
B = a_SI**2 - r_SI**2 - z_SI**2
xi = 0.5 * (-B + np.sqrt(B**2 + 4.0 * a_SI**2 * z_SI**2))

# 5. 平底円柱圧子の内部垂直変位 u_z の厳密計算
# u_z(r, z) = (2*delta / pi) * [ arctan(a / sqrt(xi)) + (a * z^2 * sqrt(xi)) / (2*(1 - nu)*(xi^2 + a^2 * z^2)) ]
sqrt_xi = np.sqrt(xi)

term1 = np.arctan(a_SI / sqrt_xi)
term2 = (a_SI * (z_SI**2) * sqrt_xi) / (2.0 * (1.0 - nu) * (xi**2 + (a_SI**2) * (z_SI**2)))

u_z_SI = (2.0 * delta_SI / np.pi) * (term1 + term2)
u_z_nm = u_z_SI * 1e9

# 6. 可視化
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

# カラーレベル基準
delta_nm_base = delta_base * 1e9
vmax_val = 1.2 * delta_nm_base
levels_setting = np.linspace(-vmax_val, vmax_val, 101)

# 'turbo' カラーマップを適用
contour = ax.contourf(R_mesh, Z_mesh, u_z_nm, levels=levels_setting, cmap='turbo', extend='both', zorder=1)

# カラーバーを下側に配置
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('Vertical Displacement, $u_z$ /nm', fontsize=11)

# 7. 平底円柱圧子の厳密接触圧力分布 p(r) = p0 / sqrt(1 - (r/a)^2) の描画
p0_scale = 5.0       # r=0 での基準高さ (nm単位, p_avg=10.0 nm相当)
p_cap = 18.5         # r -> a での無限大発散を表示範囲内に収める上限値 (nm単位)

r_plot = np.linspace(0, 30, 1000)
p_profile = np.zeros_like(r_plot)

mask_inside = r_plot < a_nm
# 1 / sqrt(1 - (r/a)^2) を計算し、グラフ上端の手前 (p_cap) でクリップ
raw_p = p0_scale / np.sqrt(np.maximum(1e-6, 1.0 - (r_plot[mask_inside] / a_nm)**2))
p_profile[mask_inside] = np.minimum(raw_p, p_cap)

ax.plot(r_plot, -p_profile, color='crimson', linewidth=1.0, ls='--', label='Flat Punch Pressure Profile', zorder=4)
ax.fill_between(r_plot, 0, -p_profile, color='crimson', alpha=0.15, zorder=3)

# 中心圧力 p0 (r=0) の補助矢印とラベル
ax.annotate('', xy=(0, 0), xytext=(0, -p0_scale),
            arrowprops=dict(facecolor='crimson', shrink=0, width=1.5, headwidth=6), zorder=4)
ax.text(0.5, -p0_scale - 2.0, '$p_0$', color='crimson', ha='left', va='center', fontsize=11, zorder=4)

# 接触端 (r=a) での応力特異性 (p -> inf) の表示
ax.text(a_nm + 0.6, -p_cap + 1.5, r'$p \to \infty$', color='crimson', ha='left', va='center', fontsize=10, zorder=4)

# 弾性体表面（z=0）の境界線
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)
# 接触ゾーン（Contact Zone）
ax.plot([0, a_nm], [0, 0], color='black', linewidth=0.5, label='Contact Zone', zorder=4)

# 領域のテキストラベル
ax.text(3, -15, 'Free Space', color='gray', fontsize=11, ha='left', va='center', zorder=3)
ax.text(3, 15, 'Elastic Body', color='black', fontsize=11, ha='left', va='center', zorder=3)

# パラメータ表示
delta_nm_current = delta_SI * 1e9
p0_MPa = p0 / 1e6
p_avg_MPa = p_avg / 1e6
param_text = (
    f"$P$ = {P*1e9:.1f} nN\n"
    f"$E$ = {E/1e6:.1f} MPa\n"
    f"$\\nu$ = {nu:.1f}\n"
    f"$a$ = {a_nm:.1f} nm\n"
    f"Calculated:\n"
    f" $\\delta$ = {delta_nm_current:.2f} nm\n"
    f" $p_0$ = {p0_MPa:.1f} MPa\n"
    f" $p_{{\\mathrm{{avg}}}}$ = {p_avg_MPa:.1f} MPa"
)
props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(17, -8, param_text, transform=ax.transData, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定
ax.set_title('Sneddon Solution (Flat Punch, Direct $u_z$)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)
ax.invert_yaxis()
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 保存
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Sneddon_flat_punch_uz_all_nu_{0:1.1f}.png'.format(nu), dpi=300, bbox_inches='tight')

plt.show()