# ヘルツ接触解（表面垂直変位u_z)

import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比
R_probe = 10.0e-9   # 球の半径: 10 nm

# 等価弾性係数 E* の計算
E_star = E / (2 * (1 - nu**2))

# ヘルツ公式より接触半径 a と全体の押し込み量 delta を算出
a = (3 * P * R_probe / (4 * E_star))**(1/3)
delta = a**2 / R_probe

# 2. 計算領域の設定 (単位: nm)
r_vec = np.linspace(-50.0, 50.0, 1000) * 1e-9
w_vec = np.zeros_like(r_vec)

# 3. 領域内（放物線）と領域外（真の厳密解）の条件分岐計算
# rではなく絶対値abs_rを使うことで、左右対称の計算を簡略化
for i, r in enumerate(r_vec):
    abs_r = np.abs(r)
    if abs_r <= a:
        w_vec[i] = delta - (abs_r**2) / (2 * R_probe)
    else:
        term1 = (2.0 - (abs_r/a)**2) * np.arcsin(a / abs_r)
        term2 = (abs_r/a) * np.sqrt(1.0 - (a / abs_r)**2)
        w_vec[i] = (delta / np.pi) * (term1 + term2)

# ナノメートル (nm) 単位に変換
r_nm = r_vec * 1e9
w_nm = w_vec * 1e9
a_nm = a * 1e9
delta_nm = delta * 1e9

# 4. グラフ描画
fig, ax = plt.subplots(figsize=(8, 6))

# 変形前の表面 (z=0 の基準線)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label=r'Undeformed Surface ($z$ = 0)')

# 変形後の表面プロファイルを描画
r_sphere_all = np.linspace(-50.0, 50.0, 1000)
# 球の放物線形状： z = delta - r^2 / 2R
z_sphere_all = delta_nm - (r_sphere_all**2) / (2 * R_probe * 1e9)
ax.plot(r_sphere_all, z_sphere_all, color='orange', linestyle='-', linewidth=1.5, label='Parabolic Probe')

# 材料の本物の表面プロファイル u_z(r)
ax.plot(r_nm, w_nm, color='g', linewidth=2.0, label=r'Displacement, $u_z(r)$')

# 重要なポイントのプロット
ax.plot(0, delta_nm, 'ro', markersize=8, label=f'Max Indentation ($\delta$ = {delta_nm:.1f} nm)')
ax.plot(a_nm, delta_nm/2, 'bo', markersize=8, label=f'Contact Edge ($r = a$, $u_z = \delta$/2)')
ax.plot(-a_nm, delta_nm/2, 'bo', markersize=8)

# グラフの見た目調整（下向き正に反転、さらに上空 z<0 が見えるようにylimの上限をマイナスまで広げる）
fig.gca().invert_yaxis()
ax.set_title('Profile of Hertzian Contact (Probe vs Elastic Body)', fontsize=12)
ax.set_xlabel(r'Radius, $r$ /nm', fontsize=12)
ax.set_ylabel(r'Displacement, $u_z(r)$ /nm', fontsize=12)

# 上空 -30.0 nm から 押し込み深さの2.0倍までを描写範囲にする
ax.set_ylim(delta_nm * 2.0, -30.0) 

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

fig.savefig('./png/Hertz_uz_surface.png', dpi=300)

plt.show()
