# ヘルツ接触解（表面垂直変位 u_z）
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
R_probe = 10.0e-9   # 球の半径: 10 nm

# 等価弾性係数 E* の計算 (剛体圧子 vs 弾性体)
# E_star = E / (1 - nu**2)
E_star = E / (1 - nu**2)

# ヘルツ公式より接触半径 a と全体の押し込み深さ delta を算出
a = (3 * P * R_probe / (4 * E_star))**(1/3)
delta = a**2 / R_probe

# 2. 計算領域の設定 (単位: m)
r_vec = np.linspace(-50.0, 50.0, 1000) * 1e-9
abs_r = np.abs(r_vec)

# 3. 領域内（放物線）と領域外（Sneddon/Hertz厳密解）のベクトル化計算
w_vec = np.zeros_like(r_vec)
inside = (abs_r <= a)
outside = ~inside

# 接触領域内 (r <= a)
w_vec[inside] = delta - (abs_r[inside]**2) / (2 * R_probe)

# 接触領域外 (r > a)
xi = a / abs_r[outside]  # a / r
term1 = (2.0 - (abs_r[outside] / a)**2) * np.arcsin(xi)
term2 = (abs_r[outside] / a) * np.sqrt(1.0 - xi**2)
w_vec[outside] = (delta / np.pi) * (term1 + term2)

# ナノメートル (nm) 単位に変換
r_nm = r_vec * 1e9
w_nm = w_vec * 1e9
a_nm = a * 1e9
delta_nm = delta * 1e9

# 4. グラフ描画
fig, ax = plt.subplots(figsize=(8, 6))

# 変形前の表面 (z=0 の基準線)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label=r'Undeformed Surface ($z = 0$)')

# 剛体球の放物線形状プロファイル： z = delta - r^2 / (2R)
r_sphere = np.linspace(-30.0, 30.0, 500)
z_sphere = delta_nm - (r_sphere**2) / (2 * R_probe * 1e9)
ax.plot(r_sphere, z_sphere, color='orange', linestyle='-', linewidth=1.5, label='Rigid Probe Profile')

# 弾性体の表面垂直変位プロファイル u_z(r)
ax.plot(r_nm, w_nm, color='green', linewidth=2.0, label=r'Displacement $u_z(r)$')

# 重要な境界点・特徴点のプロット
ax.plot(0, delta_nm, 'ro', markersize=7, label=f'Center ($\delta = {delta_nm:.2f}$ nm)')
ax.plot(a_nm, delta_nm / 2.0, 'bo', markersize=7, label=f'Edge ($r = a$, $u_z = \delta/2$)')
ax.plot(-a_nm, delta_nm / 2.0, 'bo', markersize=7)

# グラフの見た目調整（下向き変位を正とするためY軸を反転）
ax.invert_yaxis()
ax.set_title('Hertzian Contact Profile (Rigid Sphere on Elastic Half-Space)', fontsize=12)
ax.set_xlabel(r'Radius $r$ (nm)', fontsize=12)
ax.set_ylabel(r'Vertical Displacement $u_z$ (nm)', fontsize=12)

# 表示範囲の設定
ax.set_xlim(-50, 50)
ax.set_ylim(delta_nm * 2.2, -15.0)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

# ディレクトリを作成して保存
os.makedirs('./png', exist_ok=True)
fig.savefig('./png/Hertz_uz_surface.png', dpi=300)

plt.show()