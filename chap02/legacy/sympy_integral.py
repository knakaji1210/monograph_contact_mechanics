import sympy as sp

# 1. 変数と定数の定義 (すべて正の実数として定義して数学的な破綻を防ぎます)
u = sp.Symbol('u', positive=True)
a = sp.Symbol('a', positive=True)
z = sp.Symbol('z', positive=True)

# 2. 積分の中身（被積分関数）のセットアップ
# ユーザー様が提示してくださった分子着目の置換「u = a^2 - rho^2」を施した式です
integrand = (sp.sqrt(u) * (a**2 - u)) / (a**2 + z**2 - u)**(sp.Rational(5, 2))

# 前に置いておいた係数の「1/2」
factor = sp.Rational(1, 2)

# 3. 0 から a^2 までの厳密な定積分を実行
J_exact = factor * sp.integrate(integrand, (u, 0, a**2))

# 4. 結果を人間が見やすいように綺麗に整理
J_simplified = sp.simplify(J_exact)

# 5. 表示
print("--- SymPyの厳密な積分結果 ---")
sp.pprint(J_simplified)
