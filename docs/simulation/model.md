# 物理モデルと無次元化

## 基準モデル

理想的なリング・エンドキャップ型ポールトラップと、半径`R`の球形粒子を仮定する。空気抵抗は低Reynolds数でのStokes抵抗`-k v`とし、

```text
k = 6 pi eta R
m = (4/3) pi R^3 rho
```

とする。ここで`eta`は気体の粘性係数、`rho`は粒子密度である。

物理時間`t`での基準式は、符号付き半径方向座標`r`と軸方向座標`z`について

```text
m d2r/dt2 = -(Q / r0^2) (V_DC + V_AC cos(Omega t)) r - k dr/dt
m d2z/dt2 =  (2 Q / r0^2) (V_DC + V_AC cos(Omega t)) z - k dz/dt - m g
```

とする。電圧はMarch (J. Mass Spectrom. 32, 351 (1997))の規約に従い、`V_DC + V_AC cos(Omega t)`をリング電極とエンドキャップ電極の間の電位差（リング側が正）とする。このとき電位は定数項を除いて

```text
phi(r, z) = (V_DC + V_AC cos(Omega t)) (r^2 - 2 z^2) / (r0^2 + 2 z0^2)
```

であり、`r0^2 = 2 z0^2`で分母は`2 r0^2`となる（2026-10-06決定。以前の版の運動方程式は電位差`2 V`に相当する係数で、下記の`a`、`q`の定義と2倍不整合だった）。`V_AC`は振幅であり、peak-to-peak値ではない。実装時には電位と電荷の符号からこの規約を再導出し、単体テストで固定する。

電極形状は`r0^2 = 2 z0^2`を満たす理想リング・エンドキャップ型を仮定する。`r0`はリング電極内半径、`z0`は中心からエンドキャップまでの距離である。理想双曲面電極の電位はLaplace方程式から`r^2 - 2 z^2`に比例するため、`a_z = -2 a_r`、`q_z = -2 q_r`は`r0`と`z0`の比によらず成り立つ。`r0`と`z0`の比が変えるのは電圧から`a`、`q`への換算係数である。

## 無次元時間とパラメータ

```text
tau = Omega t / 2
a_z = -8 Q V_DC / (m Omega^2 r0^2)
q_z =  4 Q V_AC / (m Omega^2 r0^2)
a_r = -a_z / 2
q_r = -q_z / 2
b   =  2 k / (m Omega) = 9 eta / (R^2 rho Omega)
```

運動方程式は

```text
d2r/dtau2 + (a_r - 2 q_r cos(2 tau)) r + b dr/dtau = 0
d2z/dtau2 + (a_z - 2 q_z cos(2 tau)) z + b dz/dtau + 4 g / Omega^2 = 0
```

となる。

## 3次元モデル

理想軸対称モデルでは`x`と`y`を同じ半径方向方程式で解き、状態を

```text
[x, dx/dtau, y, dy/dtau, z, dz/dtau]
```

とする。異なる初期条件により3次元軌道を得る。`sqrt(x^2 + y^2)`を幾何学的な半径として使用し、符号付き1次元座標と混同しない。

## 安定性と捕獲可能性

重力は非同次の定数項であり、線形同次系のFloquet安定性は変えない。ただし、平衡位置、過渡応答、有限電極への衝突には影響する。

このため以下を別の結果として扱う。

- Floquet安定性: 無限時間の線形同次系が発散するか。
- 幾何学的捕獲: 指定時間内に有限電極領域から脱出しないか。
- 捕獲確率: 初期条件や物理量の分布に対する幾何学的捕獲の割合。

## 単位と入力規約

- 内部の物理量はSI単位を使用する。
- 角周波数は`rad/s`、周波数は`Hz`として別名にする。
- 状態が物理時間微分か無次元時間微分かを型またはフィールド名で区別する。
- 電荷`Q`は符号を保持する。
- `r0`と`z0`の幾何学的定義を設定ファイルに明記する。
- RF位相`rf_phase`を持つ場合、電圧は`V_DC + V_AC cos(Omega t + rf_phase)`とし、`tau = (Omega t + rf_phase) / 2`とする。既定値は0。
- 重力は`-z`方向（トラップ軸が鉛直）とする（2026-10-06確認）。

## 軌道計算の設定ファイル

`configs/*.toml`（Python標準の`tomllib`で読む）。`configs/example_microparticle.toml`は形式例であり、値は実験やノート由来ではない。

```text
[particle]   radius [m], density [kg/m^3], charge [C, 符号付き]
[gas]        viscosity [Pa s]
[trap]       r0 [m], v_dc [V], v_ac [V, 振幅], frequency [Hz], rf_phase [rad], gravity [m/s^2]
[simulation] duration [s]
[[particles]] label, position [m], velocity [m/s, 物理時間]
```
