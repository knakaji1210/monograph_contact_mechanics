# ヘルツ接触解：2つの解法の差分検証プログラム
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数および計算条件の設定 (ナノスケール)
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
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
r_vec = np.linspace(1e-5, 30, 300) # 円柱座標系 r >= 0
z_vec = np.linspace(0, 30, 300)   # 深さ 30 nm (弾性体内部のみ)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 各メッシュ点の物理量をm単位に換算 (特異点回避のため極小のオフセットを追加)
r_SI = R_mesh * 1e-9 + 1e-20
z_SI = Z_mesh * 1e-9 + 1e-20

# 4. 幾何学空間パラメータ xi (正の実根) の計算
# 解くべき方程式: xi^2 + (a^2 - r^2 - z^2)*xi - a^2*z^2 = 0
B = a_SI**2 - r_SI**2 - z_SI**2
xi = 0.5 * (-B + np.sqrt(B**2 + 4.0 * a_SI**2 * z_SI**2))

# 特異点回避コード
xi = np.maximum(xi, 1e-30)

sqrt_xi = np.sqrt(xi)
term_arctan = np.arctan(a_SI / sqrt_xi)

# -------------------------------------------------------------
# 【方法1】空間ポテンシャル Phi とその微分 dPhi_dz を経由する計算
# -------------------------------------------------------------
c = 3.0 * P / 4.0
Phi_part1 = (2.0 * a_SI**2 - r_SI**2 + 2.0 * z_SI**2) * term_arctan
Phi_part2 = a_SI * sqrt_xi * (1.0 - 3.0 * z_SI**2 / xi)
Phi = (c / a_SI**3) * (Phi_part1 + Phi_part2)

dPhi_dz = -4.0 * c * (z_SI / a_SI**3) * (a_SI / sqrt_xi - term_arctan)

u_z_SI_method1 = ((1.0 + nu) / (2.0 * np.pi * E)) * (2.0 * (1.0 - nu) * Phi - z_SI * dPhi_dz)
u_z_nm_method1 = u_z_SI_method1 * 1e9

# -------------------------------------------------------------
# 【方法2】C, f1, f2 を用いた厳密解のダイレクト計算
# -------------------------------------------------------------
C_coeff = (3.0 * (1.0 + nu) * P) / (2.0 * np.pi * (a_SI**3) * E)
f1 = ((1.0 - nu) / 2.0) * (2.0 * a_SI**2 + 2.0 * z_SI**2 - r_SI**2) - z_SI**2
f2 = ((1.0 - nu) / 2.0) * (1.0 - 3.0 * z_SI**2 / xi) + (z_SI**2 / xi)

u_z_SI_method2 = C_coeff * (f1 * term_arctan + f2 * a_SI * sqrt_xi)
u_z_nm_method2 = u_z_SI_method2 * 1e9

# -------------------------------------------------------------
# 5. 差分の評価
# -------------------------------------------------------------
u_z_diff = np.abs(u_z_nm_method1 - u_z_nm_method2)

# 6. 可視化 (テンプレート維持版)
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

# カラーレベル基準（元の u_z と同じ等高線範囲を適用）
delta_nm_base = delta_base * 1e9
vmax_val = 1.2 * delta_nm_base  
levels_setting = np.linspace(-vmax_val, vmax_val, 101)

# 差分データを元のカラープロファイルに合わせて描画
contour = ax.contourf(R_mesh, Z_mesh, u_z_diff, levels=levels_setting, cmap='turbo', extend='both', zorder=1)

# カラーバーを下側に配置
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('Absolute Difference in $u_z$ /nm', fontsize=11)

# ヘルツ面圧の描画
pressure_scale = 10.0  # 最大高さ (nm単位)
r_plot = np.linspace(0, 30, 500)
p_profile = np.zeros_like(r_plot)

mask_inside = r_plot <= a_nm
p_profile[mask_inside] = pressure_scale * np.sqrt(1.0 - (r_plot[mask_inside] / a_nm)**2)

ax.plot(r_plot, -p_profile, color='crimson', linewidth=1.0, ls='--', label='Hertz Pressure Profile', zorder=4)
ax.fill_between(r_plot, 0, -p_profile, color='crimson', alpha=0.15, zorder=3)

# 頂点への補助矢印とラベル (r=0の位置)
ax.annotate('', xy=(0, 0), xytext=(0, -pressure_scale),
            arrowprops=dict(facecolor='crimson', shrink=0, width=1.5, headwidth=6), zorder=4)
ax.text(0.5, -pressure_scale - 2, '$p_{\mathrm{max}}$', color='crimson', ha='left', va='center', fontsize=11, zorder=4)

# 弾性体表面（z=0）の境界線
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)    
# 接触ゾーン（Contact Zone）の赤線を r >= 0（0からa_nm）の範囲で描画
ax.plot([0, a_nm],[0, 0], color='black', linewidth=0.5, label='Contact Zone', zorder=4)

# 領域のテキストラベル
ax.text(3, -15, 'Free Space', color='gray', fontsize=11, ha='left', va='center', zorder=3)
ax.text(3, 15, 'Elastic Body', color='black', fontsize=11, ha='left', va='center', zorder=3)

# パラメータ表示
delta_nm_current = delta_SI * 1e9
p0_MPa = p0 / 1e6
param_text = (
    f"$P$ = {P*1e9:.1f} nN\n"
    f"$E$ = {E/1e6:.1f} MPa\n"
    f"$\\nu$ = {nu:.1f}\n"
    f"$R$ = {R_probe*1e9:.1f} nm\n"
    f"Calculated:\n"
    f" $a$ = {a_nm:.2f} nm\n"
    f" $\\delta$ = {delta_nm_current:.2f} nm\n"
    f" $p_{{\mathrm{{max}}}}$ = {p0_MPa:.1f} MPa"
)
props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(17, -8, param_text, transform=ax.transData, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定
ax.set_title('Hertzian Solution: Method Difference', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)  
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 保存
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Hertz_uz_all_diff.png', dpi=300, bbox_inches='tight')

plt.show()
