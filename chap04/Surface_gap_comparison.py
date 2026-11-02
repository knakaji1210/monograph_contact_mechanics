import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 物理パラメータ設定 (nm, nN 単位系: 1 GPa = 1 nN/nm^2, 1 J/m^2 = 1 nN/nm) ---
R = 10.0          # 球の曲率半径 [nm]
P = 1.0           # 押し込み外力 [nN]
E_star = 3.0      # 等価ヤング率 E* [GPa = nN/nm^2]
work_adh = 0.05   # 凝着仕事 Δγ [J/m^2 = nN/nm]
z0 = 0.30         # 原子間平衡距離 z_0 [nm]

# --- 2. DMT理論による接触半径 a, 押し込み量 delta, 臨界ギャップ delta_t の算出 ---
F_adh = 2.0 * np.pi * R * work_adh                    # デリャーギン凝着力 2πRΔγ [nN]
a = ((3.0 * R * (P + F_adh)) / (4.0 * E_star))**(1/3) # DMT接触半径 [nm]
delta = a**2 / R                                      # ヘルツ押し込み量 [nm]
delta_t = (9.0 * np.sqrt(3.0) / 16.0) * z0            # Dugdale臨界ギャップ (≈ 0.974 z_0) [nm]

print(f"a = {a:.2f} nm, delta = {delta:.3f} nm, delta_t = {delta_t:.3f} nm")

# --- 3. 無次元半径 x = r/a の定義 ---
x_in = np.linspace(0.0, 1.0, 100)       # 接触面内 (0 <= r/a <= 1)
x_out = np.linspace(1.0, 2.3, 1000)     # 接触面外 (r/a >= 1)

# --- 4. 接触面外 (r/a >= 1) のギャップ関数の計算 ---
# (1) 厳密なヘルツ変形ギャップ h_Hertz(r)
h_hertz = (delta / np.pi) * (
    np.sqrt(x_out**2 - 1.0) + (x_out**2 - 2.0) * np.arccos(1.0 / x_out)
)

# (2) 未変形の剛体放物面ギャップ h_rigid(r) (デリャーギン近似)
h_rigid = 0.5 * delta * (x_out**2 - 1.0)

# (3) 接触端近傍の3/2乗漸近展開 h_asymp(r)
h_asymp = (8.0 * np.sqrt(2.0) / (3.0 * np.pi)) * delta * (x_out - 1.0)**1.5

# 臨界ギャップ delta_t に達する外縁半径比 m = c/a
m_hertz = np.interp(delta_t, h_hertz, x_out)
m_rigid = np.sqrt(1.0 + 2.0 * delta_t / delta)

# --- 5. プロット作成 (単一パネル) ---
fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=120)

# 接触面内 (h = 0) と接触面外の各ギャップ曲線
ax.plot(x_in, np.zeros_like(x_in), color='black', lw=2.5, label=r'Contact region ($r \leq a$)')
ax.plot(x_out, h_hertz, color='#1f77b4', lw=2.5, label=r'Exact Hertzian gap $h_{\mathrm{Hertz}}(r)$')
ax.plot(x_out, h_rigid, color='#d62728', lw=2.0, ls='--', label=r'Rigid paraboloid $h_{\mathrm{rigid}}(r)$')
ax.plot(x_out, h_asymp, color='#2ca02c', lw=1.8, ls=':', label=r'Asymptotic $\propto (r/a - 1)^{3/2}$')

# 臨界ギャップ delta_t の水平線と外縁半径 c/a の交点
ax.axhline(delta_t, color='gray', ls='-.', lw=1.2,
           label=rf'Critical gap $\delta_t = \frac{{9\sqrt{{3}}}}{{16}}z_0 = {delta_t:.2f}\ \mathrm{{nm}}$')
ax.plot([m_hertz, m_hertz], [0, delta_t], color='#1f77b4', ls=':', lw=1.2)
ax.plot([m_rigid, m_rigid], [0, delta_t], color='#d62728', ls=':', lw=1.2)
ax.scatter([m_hertz, m_rigid], [delta_t, delta_t], color=['#1f77b4', '#d62728'], zorder=5)

# 交点の注釈
ax.annotate(rf'$c_{{\mathrm{{rigid}}}}/a = {m_rigid:.2f}$', xy=(m_rigid, delta_t),
            xytext=(m_rigid - 0.32, delta_t + 0.06), color='#d62728', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.0))
ax.annotate(rf'$c_{{\mathrm{{Hertz}}}}/a = {m_hertz:.2f}$', xy=(m_hertz, delta_t),
            xytext=(m_hertz + 0.05, delta_t - 0.08), color='#1f77b4', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.0))

ax.axvline(1.0, color='black', lw=0.8, alpha=0.4)
ax.set_xlim(0.7, 2.3)
ax.set_ylim(-0.03, 0.80)
ax.set_xlabel(r'Normalized radius, $r / a$', fontsize=12)
ax.set_ylabel(r'Surface gap $h(r)$ /nm', fontsize=12)
ax.set_title(rf'Exterior Surface Gap ($R={R:.0f}\ \mathrm{{nm}}, P={P:.1f}\ \mathrm{{nN}}, E^*={E_star:.1f}\ \mathrm{{GPa}}, a={a:.2f}\ \mathrm{{nm}}$)', fontsize=12)
ax.legend(loc='upper left', fontsize=9.5)
ax.grid(True, alpha=0.3)

# ディレクトリを作成して保存
os.makedirs('./png', exist_ok=True)
fig.savefig('./png/Surface_gap_comparison.png', dpi=300)

plt.tight_layout()
plt.show()