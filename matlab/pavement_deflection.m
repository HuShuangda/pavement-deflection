function out = pavement_deflection(layers, P_kN, r_cm, opt)
%PAVEMENT_DEFLECTION  无限大弹性层状体系：圆形均布荷载圆心弯沉
%
%   out = pavement_deflection(layers, P_kN, r_cm)
%   out = pavement_deflection(layers, P_kN, r_cm, opt)
%
% 输入
%   layers : N×3 矩阵 [h_cm, E_MPa, nu]
%              h_cm —— 层厚 (cm)；最后一行为半无限土基，填 Inf
%              E_MPa —— 回弹模量 (MPa)
%              nu    —— 泊松比 (0 < nu < 0.5)
%   P_kN   : 圆形均布荷载合力 (kN)
%   r_cm   : 荷载圆半径 (cm)，FWD 标准 φ300 mm 即 15 cm
%   opt    : 可选结构体
%              .RelTol  积分相对容差，默认 1e-11
%              .S_max   强制积分上限（无量纲 s = xi*a），默认空 = 自动
%
% 输出结构体
%   .w_001mm  圆心弯沉，单位 0.01 mm（与规范、FWD 报告口径一致）
%   .w_mm     同上，单位 mm
%   .xi_max   实际使用的积分上限 (1/mm)
%   .S        实际使用的无量纲上限 (xi_max*a)
%   .nEval    位移核被调用次数
%   .p        接地压力 (MPa)
%   .a        荷载半径 (mm)
%
% 方法
%   状态向量 X = [u_bar, w_bar, sigma_z_bar, tau_rz_bar]' 满足 dX/dz = A*X，
%   逐层传递 T_i = expm(A_i*h_i)，半无限底层用衰减子空间约束 R*X = 0 封闭，
%   表面 sigma_z_bar = -p*a*J1(xi*a)/xi、tau_rz_bar = 0。
%
%   圆心弯沉 w = c + INT_0^xi_max [ w_bar(xi)*xi - c*J1(xi*a)/xi ] dxi
%   其中 c = 2*(1-nu1^2)*p*a/E1 为顶层材料半空间的圆心沉降，即渐近减除项：
%   直接把核积分到 xi_max 会留下约 c*J0(xi_max*a) 的振荡残余（按 S^-1/2 衰减），
%   减掉再解析补回后，差值核随 exp(-2*xi*h1) 快速衰减，截断误差降到舍入量级。
%
% 单位
%   输入 cm / MPa / kN，内部统一 N-mm-MPa，输出 0.01 mm。
%
% 用法
%   参数放在 main.m 里，运行 main 即可；也可直接调用本函数。

if nargin < 3
    error('pd:args', '用法：pavement_deflection(layers, P_kN, r_cm)，层参数见 main.m。');
end
if nargin < 4 || isempty(opt), opt = struct(); end
if ~isfield(opt, 'RelTol') || isempty(opt.RelTol), opt.RelTol = 1e-11; end
if ~isfield(opt, 'S_max'),  opt.S_max = []; end

% ------------------------------------------------------------------
% 1. 输入校验与单位换算
% ------------------------------------------------------------------
if size(layers, 2) ~= 3
    error('pd:layers', 'layers 必须是 N×3 矩阵，列为 [h_cm, E_MPa, nu]。');
end
N = size(layers, 1);
h_cm = layers(:, 1);
E    = layers(:, 2);
nu   = layers(:, 3);

if N < 1
    error('pd:layers', '至少需要 1 层。');
end
if any(E <= 0)
    error('pd:E', '弹性模量必须 > 0（MPa）。');
end
if any(nu <= 0) || any(nu >= 0.5)
    error('pd:nu', '泊松比必须满足 0 < nu < 0.5。');
end
if isfinite(h_cm(N))
    error('pd:layers', '第 %d 层为底层，厚度必须填 Inf（表示半无限土基）。', N);
end
for i = 1:N-1
    if ~isfinite(h_cm(i)) || h_cm(i) <= 0
        error('pd:layers', '第 %d 层（非底层）厚度必须为有限正值，当前为 %g。', i, h_cm(i));
    end
end
if ~isscalar(P_kN) || P_kN <= 0
    error('pd:P', '荷载合力必须为 > 0 的标量 (kN)。');
end
if ~isscalar(r_cm) || r_cm <= 0
    error('pd:r', '荷载圆半径必须为 > 0 的标量 (cm)。');
end

h  = h_cm * 10;                  % cm -> mm
h(N) = Inf;
a  = r_cm * 10;                  % cm -> mm
P  = P_kN * 1000;                % kN -> N
p  = P / (pi * a^2);             % MPa = N/mm^2

G = E ./ (2 * (1 + nu));         % 剪切模量 (MPa)

% ------------------------------------------------------------------
% 2. 积分上限
% ------------------------------------------------------------------
XI_H_SAFE = 650;                 % expm 元素按 e^(xi*h) 增长，超过约 700 溢出
ASYM_FACTOR = 25;                % 差值核在 xi*h1 >= 25 时已低于 1e-9

if N > 1
    xi_max = max(ASYM_FACTOR, 8 * a / h(1)) / a;
    xi_max = min(xi_max, XI_H_SAFE / max(h(1:N-1)));
else
    xi_max = ASYM_FACTOR / a;
end
if ~isempty(opt.S_max)
    xi_max = opt.S_max / a;
end
S = xi_max * a;

% ------------------------------------------------------------------
% 3. 渐近减除项
% ------------------------------------------------------------------
c = 2 * (1 - nu(1)^2) * p * a / E(1);   % 顶层半空间圆心沉降 (mm)

% ------------------------------------------------------------------
% 4. 积分
% ------------------------------------------------------------------
nEval = 0;
AbsTol = 1e-11 * c;             % 高频段差值核已进入舍入噪声，再严只会空转

% Waypoints：每个 pi/2 处强制分段（约 1/4 个 J1 振荡周期）
dS = pi / 2;
nW = floor(S / dS - 1e-9);
if nW > 0
    wp = (1:nW) * dS / a;
else
    wp = [];
end

if isempty(wp)
    wInt = integral(@integrand, 0, xi_max, 'RelTol', opt.RelTol, 'AbsTol', AbsTol);
else
    wInt = integral(@integrand, 0, xi_max, 'RelTol', opt.RelTol, 'AbsTol', AbsTol, ...
                    'Waypoints', wp);
end

w_mm = c + wInt;

out = struct('w_001mm', w_mm * 100, 'w_mm', w_mm, ...
             'xi_max', xi_max, 'S', S, 'nEval', nEval, ...
             'p', p, 'a', a);

% ==================================================================
    function y = integrand(xi)
        % 核函数在 xi=0 处为 0/0 型，取极限 0
        nEval = nEval + numel(xi);
        y = zeros(size(xi));
        for k = 1:numel(xi)
            x = xi(k);
            if x <= 0
                y(k) = 0;
            else
                y(k) = wbar(x) * x - c * besselj(1, x * a) / x;
            end
        end
    end

    function w = wbar(xi)
        % 表面位移核 w_bar(xi)（单位压力下，mm）
        if xi < 1e-14
            sb = -p * a^2 / 2;                   % J1(x)/x -> 1/2 的极限
        else
            sb = -p * a * besselj(1, xi * a) / xi;
        end
        tb = 0;

        if N == 1
            R = decay_constraint(xi, nu(1), G(1));
            A2 = [R(1,1) R(1,2); R(2,1) R(2,2)];
            b2 = -[R(1,3) * sb + R(1,4) * tb; R(2,3) * sb + R(2,4) * tb];
            sol = A2 \ b2;
            w = sol(2);
            return
        end

        T = eye(4);
        for i = 1:N-1
            Ai = A_matrix(xi, nu(i), G(i));
            Ti = expm(Ai * h(i));
            T  = Ti * T;
            % 逐层归一化：链式乘积线性齐次，整体缩放不改变解，但可防溢出
            sc = max(max(abs(T)));
            if isfinite(sc) && sc > 1e8
                T = T / sc;
            end
        end

        R  = decay_constraint(xi, nu(N), G(N));
        RT = R * T;
        A2 = [RT(1,1) RT(1,2); RT(2,1) RT(2,2)];
        b2 = -[RT(1,3) * sb + RT(1,4) * tb; RT(2,3) * sb + RT(2,4) * tb];
        sol = A2 \ b2;
        w = sol(2);
    end
end

% ------------------------------------------------------------------
% 局部函数：状态矩阵与底部衰减约束
% ------------------------------------------------------------------
function A = A_matrix(xi, nu, G)
A = zeros(4);
A(1,2) = xi;
A(1,4) = 1 / G;
A(2,1) = -nu * xi / (1 - nu);
A(2,3) = (1 - 2 * nu) / (2 * G * (1 - nu));
A(3,4) = -xi;
A(4,1) = 2 * G * xi^2 / (1 - nu);
A(4,3) = nu * xi / (1 - nu);
end

function R = decay_constraint(xi, nu, G)
% 半无限底层衰减子空间约束（特征值 -xi 的 2x2 Jordan 块解析导出）：
%   (I)  2*G*xi*u_bar - (1-2nu)*sigma_bar + 2(1-nu)*tau_bar = 0
%   (II) 2*G*xi*w_bar + 2(1-nu)*sigma_bar - (1-2nu)*tau_bar = 0
a1 = 1 - 2 * nu;
a2 = 2 * (1 - nu);
R = zeros(2, 4);
R(1,1) = 2 * G * xi;  R(1,3) = -a1;  R(1,4) = a2;
R(2,2) = 2 * G * xi;  R(2,3) = a2;   R(2,4) = -a1;
end
