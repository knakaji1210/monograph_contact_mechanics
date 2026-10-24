# Sneddon 平底円柱圧子接触解（表面垂直変位 u_z）
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比

# 円柱圧子のパラメータ（幾何学的に固定された半径）
a_probe = 10.0e-9   # 円柱底面の半径 a: 10 nm
a = a_probe

# 等価弾性係数 E* の計算 (剛体圧子 vs 弾性体)
E_star = E / (1.0 - nu**2)

# Sneddon公式より押し込み深さ h (delta) を算出
# P = 2 * E* * a * h
delta = P / (2.0 * E_star * a)

# 2. 計算領域の設定 (単位: m)
r_vec = np.linspace(-50.0, 50.0, 1000) * 1e-9
abs_r = np.abs(r_vec)

# 3. 領域内（平坦）と領域外（Sneddon厳密解）のベクトル化計算
w_vec = np.zeros_like(r_vec)
inside = (abs_r <= a)
outside = ~inside

# 接触領域内 (r <= a): 均一な平坦変位
w_vec[inside] = delta

# 接触領域外 (r > a): Sneddon 1946 厳密解
# u_z(r) = (2 * delta / pi) * arcsin(a / r)
w_vec[outside] = (2.0 * delta / np.pi) * np.arcsin(a / abs_r[outside])

# ナノメートル (nm) 単位に変換
r_nm = r_vec * 1e9
w_nm = w_vec * 1e9
a_nm = a * 1e9
delta_nm = delta * 1e9

# 4. グラフ描画
fig, ax = plt.subplots(figsize=(8, 6))

# 変形前の表面 (z=0 の基準線)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label=r'Undeformed Surface ($z = 0$)')

# 剛体平底円柱圧子の外形線プロファイル（底面 + 側壁）
r_punch = [-a_nm, -a_nm, a_nm, a_nm]
z_punch = [-30.0, delta_nm, delta_nm, -30.0]
ax.plot(r_punch, z_punch, color='orange', linestyle='-', linewidth=2.0, label=f'Rigid Probe Profile ($a$ = {a_nm:.0f} nm)')

# 弾性体の表面垂直変位プロファイル u_z(r)
ax.plot(r_nm, w_nm, color='green', linewidth=2.0, label=r'Displacement $u_z(r)$')

# 重要な境界点・特徴点のプロット
ax.plot(0, delta_nm, 'ro', markersize=7, label=f'Center ($\\delta$ = {delta_nm:.2f} nm)')
ax.plot(a_nm, delta_nm, 'bo', markersize=7, label=f'Edge ($r = a$, $u_z = \\delta$)')
ax.plot(-a_nm, delta_nm, 'bo', markersize=7)

# グラフの見た目調整（下向き変位を正とするためY軸を反転）
ax.invert_yaxis()
ax.set_title('Sneddon Flat Punch Profile (Rigid Cylinder on Elastic Half-Space)', fontsize=12)
ax.set_xlabel(r'Radius $r$ (nm)', fontsize=12)
ax.set_ylabel(r'Vertical Displacement $u_z$ (nm)', fontsize=12)

# 表示範囲の設定
ax.set_xlim(-50, 50)
ax.set_ylim(delta_nm * 2.5, -10.0)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

# ディレクトリを作成して保存
os.makedirs('./png', exist_ok=True)
fig.savefig('./png/Sneddon_flat_punch_uz_surface.png', dpi=300)

plt.show()