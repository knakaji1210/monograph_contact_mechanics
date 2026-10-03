# ヘルツ接触解（垂直応力 σ_z）

import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数および計算条件の設定
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
R_probe = 10.0e-9   # 球の半径: 10 nm

# 接触半径 a および最大面圧 p0 の理論自動計算 ---
E_star = E / (1.0 - nu**2)                     # 等価弾性率 E*
a = ((3 * P * R_probe) / (4 * E_star))**(1/3)  # 接触半径 a の自動計算
delta = a**2 / R_probe
p0 = (3 * P) / (2 * np.pi * a**2)              # 真の最大接触面圧 p0

# 2. 計算領域の設定 (単位: nm) - テンプレートを完全踏襲
r_vec = np.linspace(-30, 30, 300)
z_vec = np.linspace(-20, 40, 300)

# グリッドの作成
R_mesh, Z_mesh = np.meshgrid(r_vec, z_vec)

# 3. 応力 σ_z の計算用配列の初期化 (初期値は NaN)
sigma_z_Pa = np.full_like(Z_mesh, np.nan)

# 4. 弾性体が存在する領域 (z >= 0) のみにヘルツ解を適用
mask_elastic = (Z_mesh >= 0)
valid_idx = mask_elastic

# 単位換算をして計算 (m単位へ)
R_SI = np.abs(R_mesh[valid_idx]) * 1e-9
Z_SI = Z_mesh[valid_idx] * 1e-9

# σ_z の計算用一時配列
sigma_z_SI = np.zeros_like(Z_SI)

# ループによる各格子点の幾何学パラメータ xi の陽解法および応力計算
for i in range(len(Z_SI)):
    r_si = R_SI[i]
    z_si = Z_SI[i]
    
    # 表面 (z = 0) の特異点・境界条件の厳密処理 (自動計算された a を適用)
    if z_si < 1e-20:
        if r_si <= a:
            sigma_z_SI[i] = p0 * np.sqrt(1.0 - (r_si / a)**2)
        else:
            sigma_z_SI[i] = 0.0
        continue

    # 2次方程式による xi の陽解法 (空間スケール基準を a に統一)
    A = 1.0
    B = a**2 - r_si**2 - z_si**2
    C = - (a**2) * (z_si**2)
    
    xi = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    # 面外極限での安全装置
    if xi <= 0.0:
        sigma_z_SI[i] = p0 if r_si <= a else 0.0
        continue
        
    # 本来の符号（素直な定義：圧縮が正）での計算
    denom = np.sqrt(xi) * (xi**2 + a**2 * z_si**2)
    sigma_z_SI[i] = p0 * (a**2 * z_si**3) / denom

# パスカル [Pa] の単位のまま格納
sigma_z_Pa[valid_idx] = sigma_z_SI

# 5. 可視化
fig, ax = plt.subplots(figsize=(8, 6))

# 応力の上限・下限（vmax_val）を設定
vmax_val = p0  # 自動計算された最大面圧 p0 までをグラデーションで表現
levels_setting = np.linspace(0, vmax_val, 51)

# 等高線プロット
contour = ax.contourf(R_mesh, Z_mesh, sigma_z_Pa, levels=levels_setting, cmap='viridis', extend='max', zorder=1)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Vertical Stress, $\sigma_z$ / Pa', fontsize=12)

# ヘルツ面圧）の重ね書き (自動計算された a と連動)
r_contact_nm = np.linspace(-a*1e9, a*1e9, 200)
r_contact_m = r_contact_nm * 1e-9

# 各位置での面圧 p(r) を計算
p_distribution = p0 * np.sqrt(1.0 - (r_contact_m / a)**2)

# 自由空間側の高さマッピング (-15nmにピークがくるようスケール調整)
z_scale_factor = -10.0 / p0 
z_profile_nm = p_distribution * z_scale_factor

# お椀型プロファイルを赤の実線で描写
ax.plot(r_contact_nm, z_profile_nm, color='red', linestyle='-', linewidth=2, zorder=4, label='Hertz Pressure Profile')
ax.fill_between(r_contact_nm, 0, z_profile_nm, color='red', alpha=0.1, zorder=2)
# --------------------------------------------------

# 弾性体との境界を線で表現
ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)

# 接触面（自動計算された -a から a）を強調表示する赤ライン
ax.plot([-a*1e9, a*1e9],[0,0], color='red', linewidth=3, zorder=4, label='Contact Area')

# 荷重点を矢印で表現
ax.annotate('', xy=(0, 0), xytext=(0, -10),
            arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8), zorder=5)
ax.text(0, -12.5, '$p_{{max}}$', color='red', ha='center', va='center', fontsize=14, zorder=5)

# 領域のテキストラベル
ax.text(-22, -5, 'Free Space', color='gray', fontsize=11, ha='center', va='center', zorder=3)
ax.text(-20, 25, 'Elastic Body', color='white', fontsize=11, ha='center', va='center', zorder=3)

# パラメータ書き込み（P, E, nu, R に加え、自動計算された a と p0 も完全出力）
param_text = (
    f"Parameters:\n"
    f" $P$ = {P*1e9:.1f} nN\n"
    f" $E$ = {E/1e6:.1f} MPa\n"
    f" $\\nu$ = {nu:.1f}\n"
    f" $R$ = {R_probe*1e9:.1f} nm\n"
    f"Calculated:\n"
    f" $a$ = {a*1e9:.2f} nm\n"
    f" $\\delta$ = {delta*1e9:.1f} nm\n"
    f" $p_{{max}}$ = {p0/1e6:.1f} MPa"
)
props = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='gray', alpha=0.9)
ax.text(15, -5, param_text, transform=ax.transData, fontsize=11,
        verticalalignment='center', horizontalalignment='left', bbox=props, zorder=5)

# 軸の設定
ax.set_title('Hertz Solution (stress, $\sigma_z$)', fontsize=14, pad=15)
ax.set_xlabel('Radius, $r$ /nm', fontsize=12)
ax.set_ylabel('Depth, $z$ /nm', fontsize=12)
ax.set_xlim(-30, 30)
ax.set_ylim(-20, 40)
ax.invert_yaxis()
ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

fig.savefig('./png/Hertz_sz.png', dpi=300)

plt.tight_layout()
plt.show()
