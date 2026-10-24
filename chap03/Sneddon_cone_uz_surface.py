# Sneddon 円錐圧子接触解（表面垂直変位 u_z）
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 物性値およびパラメータの設定
P = 1.0e-9          # 全荷重: 1 nN
E = 10.0e6          # ヤング率: 10 MPa
nu = 0.40           # ポアソン比

# 円錐圧子のパラメータ（ナノインデンターBerkovich等価円錐角: 70.3度）
alpha_deg = 60    # 円錐の半頂角 (度)
alpha = np.radians(alpha_deg)
cot_alpha = 1.0 / np.tan(alpha)

# 等価弾性係数 E* の計算 (剛体圧子 vs 弾性体)
E_star = E / (1.0 - nu**2)

# Sneddon公式より接触半径 a と全体の押し込み深さ h (delta) を算出
# P = (pi / 2) * E* * a^2 * cot(alpha)
a = np.sqrt(2.0 * P / (np.pi * E_star * cot_alpha))
delta = (np.pi / 2.0) * a * cot_alpha  # h = (pi/2) * a * cot(alpha)

# 2. 計算領域の設定 (単位: m)
r_vec = np.linspace(-50.0, 50.0, 1000) * 1e-9
abs_r = np.abs(r_vec)

# 3. 領域内（円錐）と領域外（Sneddon厳密解）のベクトル化計算
w_vec = np.zeros_like(r_vec)
inside = (abs_r <= a)
outside = ~inside

# 接触領域内 (r <= a): 圧子形状そのまま
w_vec[inside] = delta - abs_r[inside] * cot_alpha

# 接触領域外 (r > a): Sneddon 1965 厳密解
# u_z(r) = (2 * delta / pi) * [ arcsin(a/r) - (r/a) + sqrt((r/a)^2 - 1) ]
x_out = abs_r[outside] / a   # 無次元半径 x = r / a
term_arcsin = np.arcsin(1.0 / x_out)
term_cone = -x_out + np.sqrt(x_out**2 - 1.0)
w_vec[outside] = (2.0 * delta / np.pi) * (term_arcsin + term_cone)

# ナノメートル (nm) 単位に変換
r_nm = r_vec * 1e9
w_nm = w_vec * 1e9
a_nm = a * 1e9
delta_nm = delta * 1e9

# 接触端 (r = a) における変位: u_z(a) = (1 - 2/pi) * delta ≈ 0.363 * delta
w_edge_nm = (1.0 - 2.0 / np.pi) * delta_nm

# 4. グラフ描画
fig, ax = plt.subplots(figsize=(8, 6))

# 変形前の表面 (z=0 の基準線)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label=r'Undeformed Surface ($z = 0$)')

# 剛体円錐圧子のプロファイル: z = delta - |r| * cot(alpha)
r_cone = np.linspace(-30.0, 30.0, 500)
z_cone = delta_nm - np.abs(r_cone) * cot_alpha
ax.plot(r_cone, z_cone, color='orange', linestyle='-', linewidth=1.5, label=f'Rigid Probe Profile ($\\alpha$ = {alpha_deg}°)')

# 弾性体の表面垂直変位プロファイル u_z(r)
ax.plot(r_nm, w_nm, color='green', linewidth=2.0, label=r'Displacement $u_z(r)$')

# 重要な境界点・特徴点のプロット
ax.plot(0, delta_nm, 'ro', markersize=7, label=f'Apex ($\\delta$ = {delta_nm:.2f} nm)')
ax.plot(a_nm, w_edge_nm, 'bo', markersize=7, label=f'Edge ($r = a$, $u_z = 0.18\delta$)')
ax.plot(-a_nm, w_edge_nm, 'bo', markersize=7)

# グラフの見た目調整（下向き変位を正とするためY軸を反転）
ax.invert_yaxis()
ax.set_title('Sneddon Conical Contact Profile (Rigid Cone on Elastic Half-Space)', fontsize=12)
ax.set_xlabel(r'Radius $r$ (nm)', fontsize=12)
ax.set_ylabel(r'Vertical Displacement $u_z$ (nm)', fontsize=12)

# 表示範囲の設定
ax.set_xlim(-50, 50)
ax.set_ylim(delta_nm * 2.2, -15.0)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=10)

# ディレクトリを作成して保存
os.makedirs('./png', exist_ok=True)
fig.savefig('./png/Sneddon_cone_uz_surface.png', dpi=300)

plt.show()