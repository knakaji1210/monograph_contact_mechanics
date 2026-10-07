# ヘルツ接触解（円柱座標系 r >= 0 におけるラジアル応力 σ_r）

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
r_vec = np.linspace(1e-5, 30, 300) # テンプレート基準：円柱座標系 r >= 0
z_vec = np.linspace(0, 30, 300)   # テンプレート基準：深さ 30 nm (弾性体内部のみ)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 各メッシュ点の物理量をm単位に換算 (ゼロ割り・特異点回避のため極小のオフセット 1e-20 を追加)
r_SI = R_mesh * 1e-9 + 1e-20
z_SI = Z_mesh * 1e-9 + 1e-20

# 4. 幾何学空間パラメータ xi (正の実根) の計算
B = a_SI**2 - r_SI**2 - z_SI**2
xi = 0.5 * (-B + np.sqrt(B**2 + 4 * a_SI**2 * z_SI**2))
xi_safe = np.where(xi < 1e-20, 1e-20, xi) # ゼロ割り防御壁
sqrt_xi = np.sqrt(xi_safe)

# 5. ラジアル応力 σ_r の計算
# f_1: 遠方独立項
f_1 = ((1.0 - 2.0 * nu) * a_SI**3) / (3.0 * r_SI**2) * (1.0 - (z_SI**3) / (xi_safe * sqrt_xi))

# f_2: 変位勾配骨格項
f_2 = (1.0 + nu) * z_SI * (np.arctan(a_SI / sqrt_xi) - a_SI / sqrt_xi)

# f_3: 空間幾何調和項
f_3 = (nu * a_SI**3 * z_SI) / (sqrt_xi * (a_SI**2 + xi_safe))

# f_4: 幾何最深部有理式項
f_4 = - (z_SI * a_SI**3 * sqrt_xi * (xi_safe - z_SI**2)) / ((xi_safe**2 + a_SI**2 * z_SI**2) * (a_SI**2 + xi_safe))

# 共通係数を掛け合わせて内部応力場(Pa)を算出
common_factor = (3.0 * P) / (2.0 * np.pi * a_SI**3)
sigma_r_internal = common_factor * (f_1 + f_2 + f_3 + f_4)

# 最表面 (z=0) の境界条件を解析的厳密解へ滑らかに差し替え
sigma_r_inside = -p0 * (np.sqrt(1.0 - (r_SI / a_SI)**2) - ((1.0 - 2.0 * nu) / 3.0) * (a_SI**2 / r_SI**2) * (1.0 - (1.0 - (r_SI / a_SI)**2)**1.5))
sigma_r_outside = p0 * (1.0 - 2.0 * nu) / 3.0 * (a_SI**2 / r_SI**2)

sigma_r_surface = np.where(r_SI <= a_SI, sigma_r_inside, sigma_r_outside)
sigma_r_Pa = np.where(z_SI < 1e-11, sigma_r_surface, sigma_r_internal)

# 表示用に MPa 単位へ換算
sigma_r_MPa = sigma_r_Pa / 1e6

# 6. 可視化
fig, ax = plt.subplots(figsize=(4.8, 6.4))
ax.set_aspect('equal')

# 💥【センターゼロ・正負対称等高線ロジック】
p0_MPa = p0 / 1e6
vmax_val = p0_MPa
levels_setting = np.linspace(-vmax_val, vmax_val, 101)

# 'turbo' カラーマップを厳格に適用
contour = ax.contourf(R_mesh, Z_mesh, sigma_r_MPa, levels=levels_setting, cmap='turbo', extend='both', zorder=1)

# 正負が切り替わる「ゼロ境界線」のマッピング ───
# レベル0の位置に、太さ0.5の黒いダッシュ線を引く。凡例にも登録。
boundary = ax.contour(R_mesh, Z_mesh, sigma_r_MPa, levels=[0.0], colors='black', linewidths=0.5, linestyles='--', zorder=5)
# 凡例表示用にダミーの線を登録するための処理
ax.plot([], [], color='black', linewidth=1.5, linestyle='--', label=r'Boundary ($\sigma_r = 0$)')

# カラーバーを下側に配置
cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.08, aspect=25)
cbar.set_label('Radial Stress, $\sigma_r$ /MPa', fontsize=11)

# ヘルツ面圧の描画
pressure_scale = 10.0  
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
# 接触ゾーン（Contact Zone）の黒線
ax.plot([0, a_nm], [0, 0], color='black', linewidth=0.5, label='Contact Zone', zorder=4)

# 領域のテキストラベル
ax.text(3, -15, 'Free Space', color='gray', fontsize=11, ha='left', va='center', zorder=3)
ax.text(3, 15, 'Elastic Body', color='black', fontsize=11, ha='left', va='center', zorder=3)

# パラメータ表示
delta_nm_current = delta_SI * 1e9
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
ax.set_title('Hertzian Solution ($\sigma_r$)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-20, 30)  
ax.invert_yaxis()  
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

# 保存
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Hertz_sr.png', dpi=300, bbox_inches='tight')

plt.show()
