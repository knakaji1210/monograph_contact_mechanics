import numpy as np
import matplotlib.pyplot as plt

# 1. 物理定数の設定 (ナノスケール)
P = 10.0e-9        # 荷重: 10 nN
E = 1.0e9          # ヤング率: 1.0 GPa
nu = 0.40          # ポアソン比

# 2. 計算グリッドの作成 (横方向 r, 深さ方向 z ともに 0.05 nm 〜 10 nm)
# ※原点 (0,0) の特異点（無限大）を避けるため、わずかにずらしてスタートします
r_vec = np.linspace(0.05, 10, 300) * 1e-9  # メートル換算
z_vec = np.linspace(0.05, 10, 300) * 1e-9  # メートル換算
R_grid, Z_grid = np.meshgrid(r_vec, z_vec)

# 原点からの直線距離 R = sqrt(r^2 + z^2) の計算
R = np.sqrt(R_grid**2 + Z_grid**2)

# 3. ブジネスク公式による応力計算 (単位: Pa)
sigma_z = (3 * P / (2 * np.pi)) * (Z_grid**3 / R**5)      # 垂直応力
tau_rz = (3 * P / (2 * np.pi)) * (R_grid * Z_grid**2 / R**5) # せん断応力

# パスカル(Pa) から ギガパスカル(GPa) へ単位変換
sigma_z_GPa = sigma_z * 1e-9
tau_rz_GPa = tau_rz * 1e-9

# 4. グラフ描画 (2画面並べてプロット)
fig, axes = plt.subplots(1, 2, figsize=(14,6),tight_layout=True)
fig.suptitle(r'Boussinesq Stress Distribution under Point Load ($P$ = {0:.1f} nN, $E$ = {1:.1f} GPa, $\nu$ = {2:.2f})'.format(P * 1e9, E * 1e-9, nu), fontsize=16)

# 左側：垂直応力 sigma_z のカラーマップ
im0 = axes[0].imshow(sigma_z_GPa, extent=[0, 10, 10, 0], cmap='jet', aspect='auto', vmax=1.5)
axes[0].set_title('Vertical Stress, $\sigma_z$ /GPa', fontsize=14)
axes[0].set_xlabel('Radius, $r$ /nm', fontsize=12)
axes[0].set_ylabel('Depth, $z$ /nm', fontsize=12)
fig.colorbar(im0, ax=axes[0], label='Stress /GPa')

# 右側：せん断応力 tau_rz のカラーマップ
im1 = axes[1].imshow(tau_rz_GPa, extent=[0, 10, 10, 0], cmap='jet', aspect='auto', vmax=0.5)
axes[1].set_title('Shear Stress, $\\tau_{rz}$ /GPa', fontsize=14)
axes[1].set_xlabel('Radius, $r$ /nm', fontsize=12)
axes[1].set_ylabel('Depth, $z$ /nm', fontsize=12)
fig.colorbar(im1, ax=axes[1], label='Stress /GPa')

fig.savefig('./png/Boussinesq_Stress_Distribution.png', dpi=300)

plt.tight_layout()
plt.show()
