# ヘルツ接触解（垂直変位u_z)

import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数および計算条件の設定 (ナノスケール)
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.20           # ポアソン比
R_probe = 10.0e-9   # 球の半径: 10 nm

# 2. 絶対的な等高線基準の計算（nu=0.5 の delta を絶対基準として固定）
nu_base = 0.5
E_star_base = E / (1.0 - nu_base**2)
a_base = (3.0 * P * R_probe / (4.0 * E_star_base))**(1.0 / 3.0)
delta_base = a_base**2 / R_probe

# 現在のポアソン比 nu における実際の弾性諸元
E_star = E / (1.0 - nu**2)
a_SI = (3.0 * P * R_probe / (4.0 * E_star))**(1.0 / 3.0)
delta_SI = a_SI**2 / R_probe
p0 = 3.0 * P / (2.0 * np.pi * a_SI**2)

# 3. 計算領域の設定 (単位: nm)
a_nm = a_SI * 1e9
r_vec = np.linspace(-30, 30, 300)
z_vec = np.linspace(-20, 40, 300)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 4. 変位 u_z の計算用配列の初期化
u_z_nm = np.full_like(Z_mesh, np.nan)
mask_elastic = (Z_mesh >= 0)

# 各メッシュ点の物理量をm単位に換算 (ゼロ割り・特異点回避のため極小のオフセット 1e-20 を追加)
r_SI = np.abs(R_mesh[mask_elastic]) * 1e-9 + 1e-20
z_SI = Z_mesh[mask_elastic] * 1e-9 + 1e-20

# 5. 幾何学空間パラメータ xi (正の実根) の計算
B = a_SI**2 - r_SI**2 - z_SI**2
xi = 0.5 * (-B + np.sqrt(B**2 + 4 * a_SI**2 * z_SI**2))

# 6. 数学パーツの準備
sqrt_xi = np.sqrt(xi)

# 共通パーツ arctan(a / sqrt(xi)) の処理
arg_arctan = a_SI / sqrt_xi
term_arctan = np.arctan(arg_arctan)

# 7. 「真の空間ポテンシャル Phi」の計算
c = 3.0 * P / 4.0
Phi_part1 = (2.0 * a_SI**2 - r_SI**2 + 2.0 * z_SI**2) * term_arctan
Phi_part2 = a_SI * sqrt_xi * (1.0 - 3.0 * z_SI**2 / xi)
Phi = (c / a_SI**3) * (Phi_part1 + Phi_part2)

# 8. 「厳密偏微分 dPhi_dz」の計算
dPhi_dz = -4.0 * c * (z_SI / a_SI**3) * (a_SI / sqrt_xi - term_arctan)

# 9. 垂直変位 u_z の最終計算
u_z_SI = ((1.0 + nu) / (2.0 * np.pi * E)) * (2.0 * (1.0 - nu) * Phi - z_SI * dPhi_dz)
u_z_nm[mask_elastic] = u_z_SI * 1e9

# 10. 可視化
fig, ax = plt.subplots(figsize=(8, 6))

# カラーレベルの基準を完全に固定し、絶対評価を断行
delta_nm_base = delta_base * 1e9
vmax_val = 1.2 * delta_nm_base  
vmin_val = -0.05 * delta_nm_base 
levels_setting = np.linspace(vmin_val, vmax_val, 51)

# 変位場の等高線プロット
contour = ax.contourf(R_mesh, Z_mesh, u_z_nm, levels=levels_setting, cmap='viridis', extend='both', zorder=1)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Vertical Displacement, $u_z$ /nm', fontsize=12)

# ヘルツ面圧（お椀型プロファイル）の描画
pressure_scale = 10.0  # お椀のグラフ上の最大高さ (nm単位)
r_plot = np.linspace(-30, 30, 500)
p_profile = np.zeros_like(r_plot)

mask_inside = np.abs(r_plot) <= a_nm
p_profile[mask_inside] = pressure_scale * np.sqrt(1.0 - (r_plot[mask_inside] / a_nm)**2)

ax.plot(r_plot, -p_profile, color='crimson', linewidth=2.5, label='Hertz Pressure Profile', zorder=4)
ax.fill_between(r_plot, 0, -p_profile, color='crimson', alpha=0.15, zorder=3)

# 頂点への補助矢印とラベル
ax.annotate('', xy=(0, 0), xytext=(0, -pressure_scale),
            arrowprops=dict(facecolor='crimson', shrink=0, width=1.5, headwidth=6), zorder=4)
ax.text(0, -pressure_scale - 2, '$p_{{max}}$', color='crimson', ha='center', va='center', fontsize=12, zorder=4)

# 境界線の描画
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)    
ax.plot([-a_nm, a_nm], [0, 0], color='red', linewidth=4.0, label='Contact Zone', zorder=4)

ax.text(-20, -15, 'Free Space', color='gray', fontsize=11, ha='center', va='center', zorder=3)
ax.text(-20, 25, 'Elastic Body', color='white', fontsize=11, ha='center', va='center', zorder=3)

# パラメータ表示
delta_nm_current = delta_SI * 1e9
p0_MPa = p0 / 1e6
param_text = (
    f"Parameters:\n"
    f" $P$ = {P*1e9:.1f} nN\n"
    f" $E$ = {E/1e6:.1f} MPa\n"
    f" $\\nu$ = {nu:.1f}\n"
    f" $R$ = {R_probe*1e9:.1f} nm\n"
    f"Calculated:\n"
    f" $a$ = {a_nm:.2f} nm\n"
    f" $\\delta$ = {delta_nm_current:.2f} nm\n"
    f" $p_{{max}}$ = {p0_MPa:.1f} MPa"
)
props = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(15, -2, param_text, transform=ax.transData, fontsize=10, bbox=props, zorder=5)

# 軸の設定
ax.set_title('Hertzian Solution (displacement, $u_z$)', fontsize=14, pad=15)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 40)
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

fig.savefig('./png/Hertz_uz_all_nu_{0:1.1f}.png'.format(nu), dpi=300)

plt.tight_layout()
plt.show()
