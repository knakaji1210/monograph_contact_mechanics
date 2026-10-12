import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数および計算条件の設定 (ナノスケール基準)
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
R_probe = 10.0e-9   # 球の半径: 10 nm

# 現在の弾性諸元および接触パラメータの計算
E_star = E / (1.0 - nu**2)
a_SI = (3.0 * P * R_probe / (4.0 * E_star))**(1.0 / 3.0)
a_nm = a_SI * 1e9
p0 = 3.0 * P / (2.0 * np.pi * a_SI**2)
p0_MPa = p0 / 1e6

# 2. プロット用の半径軸の設定 (0 から 30 nm まで)
r_plot_nm = np.linspace(1e-5, 30, 1000)
r_plot_m = r_plot_nm * 1e-9

# 各領域のマジックマスク作成
mask_inside = r_plot_m <= a_SI
mask_outside = r_plot_m > a_SI

# 3. 各応力成分の最表面 (z=0) 厳密解の計算 (Pa単位)
# --- 垂直応力 sigma_z ---
sigma_z_surface = np.zeros_like(r_plot_m)
sigma_z_surface[mask_inside] = -p0 * np.sqrt(1.0 - (r_plot_m[mask_inside] / a_SI)**2)
# mask_outside は 0 のまま維持

# --- ラジアル応力 sigma_r ---
sigma_r_surface = np.zeros_like(r_plot_m)
# 接触面内 (r <= a)
r_in = r_plot_m[mask_inside]
term_ratio_in = (a_SI**2 / r_in**2) * (1.0 - (1.0 - (r_in / a_SI)**2)**1.5)
sigma_r_surface[mask_inside] = -p0 * (np.sqrt(1.0 - (r_in / a_SI)**2) - ((1.0 - 2.0 * nu) / 3.0) * term_ratio_in)
# 接触面外 (r > a)
r_out = r_plot_m[mask_outside]
sigma_r_surface[mask_outside] = p0 * ((1.0 - 2.0 * nu) / 3.0) * (a_SI**2 / r_out**2)

# --- 周方向応力 sigma_theta ---
sigma_theta_surface = np.zeros_like(r_plot_m)
# 💥 核心部修正：謎様が看破した真の厳密解 (第1項へ -2*nu を正しく分配)
sigma_theta_surface[mask_inside] = p0 * (-2.0 * nu * np.sqrt(r_in / a_SI * 0.0 + 1.0 - (r_in / a_SI)**2) - ((1.0 - 2.0 * nu) / 3.0) * term_ratio_in)
# 接触面外 (r > a)
sigma_theta_surface[mask_outside] = -p0 * ((1.0 - 2.0 * nu) / 3.0) * (a_SI**2 / r_out**2)

# --- 剪断応力 tau_rz ---
# 表面全域で恒等的にゼロ
tau_rz_surface = np.zeros_like(r_plot_m)

# 4. MPa 単位への換算
sz_MPa = sigma_z_surface / 1e6
sr_MPa = sigma_r_surface / 1e6
st_MPa = sigma_theta_surface / 1e6
trz_MPa = tau_rz_surface / 1e6

# 5. 二次元プロットの構築
fig, ax = plt.subplots(figsize=(6.4, 4.8))  # プロファイル用の横長標準アスペクト

# 各応力曲線の描画
ax.plot(r_plot_nm, sz_MPa, color='black', linewidth=2.0, label=r'$\sigma_z$ (Vertical)')
ax.plot(r_plot_nm, sr_MPa, color='crimson', linewidth=2.0, label=r'$\sigma_r$ (Radial)')
ax.plot(r_plot_nm, st_MPa, color='royalblue', linewidth=2.0, label=r'$\sigma_\theta$ (Circumferential)')
ax.plot(r_plot_nm, trz_MPa, color='gray', linewidth=1.5, linestyle=':', label=r'$\tau_{rz}$ (Shear)')

# 💥 視覚的ハイライト：接触円エッジ (r = a) の垂直境界ダッシュ線
ax.axvline(a_nm, color='darkgray', linewidth=1.0, linestyle='--', zorder=0)
ax.text(a_nm + 0.5, p0_MPa * 0.2, f'Edge $r=a$\n({a_nm:.2f} nm)', color='darkgray', fontsize=10, va='bottom')

# 軸の基準線と装飾
ax.axhline(0, color='black', linewidth=1.0, linestyle='-', zorder=0)
ax.set_title('Surface Stress Profiles at $z=0$ (Hertz Solution)', fontsize=13, pad=12)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Stress, $\sigma$ /MPa', fontsize=12)
ax.set_xlim(0, 30)
ax.set_ylim(-p0_MPa * 1.2, p0_MPa * 0.5)  # 圧縮メインのため下側を広く確保
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')

# パラメータボックスの配置
param_text = (
    f"$P$ = {P*1e9:.1f} nN\n"
    f"$E$ = {E/1e6:.1f} MPa\n"
    f"$\\nu$ = {nu:.1f}\n"
    f"$R$ = {R_probe*1e9:.1f} nm\n"
    f"Calculated:\n"
    f" $a$ = {a_nm:.2f} nm\n"
    f" $p_{{max}}$ = {p0_MPa:.1f} MPa"
)
props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(0.75, 0.05, param_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='left', bbox=props, zorder=5)

# 保存および表示
os.makedirs('./png', exist_ok=True)
plt.tight_layout()
fig.savefig('./png/Hertz_Surface_Stresses.png', dpi=300, bbox_inches='tight')
plt.show()
