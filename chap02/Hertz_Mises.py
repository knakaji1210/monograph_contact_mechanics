# ヘルツ接触解（円柱座標系 r >= 0 における von Mises 等価応力）

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
r_vec = np.linspace(0, 30, 300)   # r=0 を含む厳密なグリッド
z_vec = np.linspace(0, 30, 300)   # 深さ 30 nm 固定

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 各メッシュ点の物理量をm単位に換算
r_SI = R_mesh * 1e-9
z_SI = Z_mesh * 1e-9

# 4. 幾何学空間パラメータ xi (正の実根) の計算
r_safe_xi = np.where((r_SI == 0) & (z_SI == 0), 1e-20, r_SI)
z_safe_xi = np.where((r_SI == 0) & (z_SI == 0), 1e-20, z_SI)

B = a_SI**2 - r_safe_xi**2 - z_safe_xi**2
xi = 0.5 * (-B + np.sqrt(B**2 + 4.0 * a_SI**2 * z_safe_xi**2))
xi_safe = np.where(xi < 1e-25, 1e-25, xi) # ゼロ割り防御壁
sqrt_xi = np.sqrt(xi_safe)

# 5. 4応力成分の厳密内部計算 (Pa単位)
common_factor = (3.0 * P) / (2.0 * np.pi * a_SI**3)

# ① 垂直応力 σ_z (代数形式はプラスとして計算)
denom_z = sqrt_xi * (xi_safe**2 + a_SI**2 * z_safe_xi**2)
sigma_z = p0 * (a_SI**2 * z_safe_xi**3) / denom_z

# ② ラジアル応力 σ_r (4分割型)
r_div = np.where(r_SI < 1e-15, 1e-15, r_SI)
f_1 = ((1.0 - 2.0 * nu) * a_SI**3) / (3.0 * r_div**2) * (1.0 - (z_safe_xi**3) / (xi_safe * sqrt_xi))
f_1 = np.where(R_mesh == 0, 0.0, f_1) # 💡 中心軸上の数値誤差バーストを厳密ゼロ化防壁で迎撃
f_2 = (1.0 + nu) * z_safe_xi * (np.arctan(a_SI / sqrt_xi) - a_SI / sqrt_xi)
f_3 = (nu * a_SI**3 * z_safe_xi) / (sqrt_xi * (a_SI**2 + xi_safe))
f_4 = - (z_safe_xi * a_SI**3 * sqrt_xi * (xi_safe - z_safe_xi**2)) / ((xi_safe**2 + a_SI**2 * z_safe_xi**2) * (a_SI**2 + xi_safe))
sigma_r = common_factor * (f_1 + f_2 + f_3 + f_4)

# ③ 周方向応力 σ_θ ((1-ν)括り出し3大パーツ型)
g_1 = - f_1
g_2 = f_2
g_3 = ((1.0 - nu) * a_SI**3 * z_safe_xi) / (sqrt_xi * (a_SI**2 + xi_safe))
sigma_theta = common_factor * (g_1 + g_2 + g_3)

# ④ 剪断応力 τ_rz (幾何核集中型)
denom_tau = (xi_safe**2 + a_SI**2 * z_safe_xi**2) * (a_SI**2 + xi_safe)
h_1 = - (r_safe_xi * a_SI**3 * (z_safe_xi**2) * sqrt_xi) / denom_tau
tau_rz = common_factor * h_1

# ─── 最表面(z=0)の解析的極限上書き ───
mask_surface = (Z_mesh == 0)
idx_inside = (R_mesh <= a_nm) & mask_surface
r_inside_SI = r_SI[idx_inside]
sigma_r_surf_inside = np.zeros_like(r_inside_SI)

r_zero_mask = (r_inside_SI == 0)
sigma_r_surf_inside[r_zero_mask] = -p0 * (0.5 + nu)
r_nonzero = r_inside_SI[~r_zero_mask]
sigma_r_surf_inside[~r_zero_mask] = -p0 * (np.sqrt(1.0 - (r_nonzero / a_SI)**2) - 
    ((1.0 - 2.0 * nu) / 3.0) * (a_SI**2 / r_nonzero**2) * (1.0 - (1.0 - (r_nonzero / a_SI)**2)**1.5))

idx_outside = (R_mesh > a_nm) & mask_surface
sigma_r_surf_outside = p0 * (1.0 - 2.0 * nu) / 3.0 * (a_SI**2 / r_SI[idx_outside]**2)

sigma_z[mask_surface] = np.where(R_mesh[mask_surface] <= a_nm, p0 * np.sqrt(1.0 - (r_SI[mask_surface] / a_SI)**2), 0.0)
sigma_r[idx_inside] = sigma_r_surf_inside
sigma_r[idx_outside] = sigma_r_surf_outside
sigma_theta[idx_inside] = sigma_r_surf_inside
sigma_theta[idx_outside] = -sigma_r_surf_outside
tau_rz[mask_surface] = 0.0

# 中心軸上 (r = 0) の等方バインドの完全維持
mask_axis = (R_mesh == 0)
sigma_theta[mask_axis] = sigma_r[mask_axis]
tau_rz[mask_axis] = 0.0

# σ_z を物理的な真の符号である「圧縮＝負」に反転させてからMisesのブラケットに流し込む！
sigma_z_physical = -sigma_z
sigma_z_physical[mask_surface] = -np.where(R_mesh[mask_surface] <= a_nm, p0 * np.sqrt(1.0 - (r_SI[mask_surface] / a_SI)**2), 0.0)

mises_bracket = (sigma_r - sigma_theta)**2 + (sigma_theta - sigma_z_physical)**2 + (sigma_z_physical - sigma_r)**2
sigma_mises_Pa = np.sqrt(0.5 * mises_bracket + 3.0 * tau_rz**2)
sigma_mises_MPa = sigma_mises_Pa / 1e6

# 7. 可視化
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

p0_MPa = p0 / 1e6
vmax_val = p0_MPa
levels_setting = np.linspace(0.0, vmax_val, 101)

contour = ax.contourf(R_mesh, Z_mesh, sigma_mises_MPa, levels=levels_setting, cmap='turbo', extend='max', zorder=1)

# 内部ピーク
max_idx = np.unravel_index(np.argmax(sigma_mises_MPa), sigma_mises_MPa.shape)
r_peak_nm = R_mesh[max_idx]
z_peak_nm = Z_mesh[max_idx]
mises_max_val = sigma_mises_MPa[max_idx]
ax.scatter(r_peak_nm, z_peak_nm, color='white', marker='o', s=60, edgecolors='black', linewidths=1.0, zorder=5, label='Yield Initiation Peak')

# カラーバーを下側に配置
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('von Mises Equivalent Stress, $\sigma_{\mathrm{Mises}}$ /MPa', fontsize=11)

# ヘルツ面圧の描画
pressure_scale = 10.0  
r_plot = np.linspace(0, 30, 500)
p_profile = np.zeros_like(r_plot)

mask_inside_plot = r_plot <= a_nm
p_profile[mask_inside_plot] = pressure_scale * np.sqrt(1.0 - (r_plot[mask_inside_plot] / a_nm)**2)

ax.plot(r_plot, -p_profile, color='crimson', linewidth=1.0, ls='--', label='Hertz Pressure Profile', zorder=4)
ax.fill_between(r_plot, 0, -p_profile, color='crimson', alpha=0.15, zorder=3)

# 頂点への補助矢印とラベル
ax.annotate('', xy=(0, 0), xytext=(0, -pressure_scale),
            arrowprops=dict(facecolor='crimson', shrink=0, width=1.5, headwidth=6), zorder=4)
ax.text(0.5, -pressure_scale - 2, '$p_{\mathrm{max}}$', color='crimson', ha='left', va='center', fontsize=11, zorder=4)

ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)    
ax.plot([0, a_nm], [0, 0], color='black', linewidth=0.5, label='Contact Zone', zorder=4)

ax.text(3, -15, 'Free Space', color='gray', fontsize=11, ha='left', va='center', zorder=3)
ax.text(3, 15, 'Elastic Body', color='black', fontsize=11, ha='left', va='center', zorder=3)

# パラメータ表示 (Mises最大値情報を追記)
delta_nm_current = delta_SI * 1e9
param_text = (
    f"$P$ = {P*1e9:.1f} nN\n"
    f"$E$ = {E/1e6:.1f} MPa\n"
    f"$\\nu$ = {nu:.1f}\n"
    f"$R$ = {R_probe*1e9:.1f} nm\n"
    f"Calculated:\n"
    f" $a$ = {a_nm:.2f} nm\n"
    f" $\\delta$ = {delta_nm_current:.2f} nm\n"
    f" $p_{{\mathrm{{max}}}}$ = {p0_MPa:.1f} MPa\n"
    f"Mises Peak:\n"
    f" $\sigma_{{\mathrm{{max}}}}$ = {mises_max_val:.1f} MPa\n"
    f" at $z$ = {z_peak_nm:.2f} nm"
)
props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(17, -10, param_text, transform=ax.transData, fontsize=10,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

ax.set_title('Hertzian Solution (von Mises Stress)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)  
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Hertz_Mises.png', dpi=300, bbox_inches='tight')

plt.show()
