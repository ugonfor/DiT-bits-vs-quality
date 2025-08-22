# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: (H, W) 2D matrix (can be on CPU or CUDA)
    :param    reduced_rank: target rank r
    :return: L (H, r), R (r, W)
    """
    assert weight.dim() == 2, "Only Support 2D matrix"

    H, W = weight.shape
    r = int(reduced_rank)
    assert 1 <= r <= min(H, W), f"reduced_rank must be in [1, {min(H, W)}]"

    # keep original device/dtype
    orig_device = weight.device
    orig_dtype = weight.dtype

    # ---- run SVD on GPU in float32 if possible ----
    # (move only for the SVD, then move results back)
    run_device = torch.device("cuda") if torch.cuda.is_available() else orig_device
    W32 = weight.to(run_device, dtype=torch.float32, copy=(run_device != orig_device or weight.dtype != torch.float32))

    # cuSOLVER backend (GPU) or MAGMA/LAPACK (CPU)
    # full_matrices=False gives thin SVD
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)

    # Truncate to rank r
    U = U[:, :r]              # (H, r)
    S = S[:r]                 # (r,)
    Vh = Vh[:r, :]            # (r, W)

    # Build L, R without forming diag(S): cheaper & numerically nicer
    sqrtS = torch.sqrt(S)     # (r,)
    L = U * sqrtS             # (H, r)  broadcast over columns
    R = sqrtS.unsqueeze(1) * Vh  # (r, 1) * (r, W) -> (r, W)

    # Move back to original device/dtype
    L = L.to(device=orig_device, dtype=orig_dtype)
    R = R.to(device=orig_device, dtype=orig_dtype)
    return L, R



class LinearQuant:
    def __init__(self, tensor, scale, zero_point, n_bits, layerwise=False):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.n_bits = n_bits
        self.layerwise = layerwise

    def __call__(self, tensor, scale, zero_point, n_bits, layerwise=False):
        return self.quantize_dequantize(tensor, scale, zero_point, n_bits, layerwise)

    def quantize_dequantize(self, tensor, scale, zero_point, n_bits, layerwise=False):
        # Calculate quantization range
        qmin = 0
        qmax = 2**n_bits - 1

        if layerwise:
            # Use single scale and zero_point for entire tensor
            scale = scale.view(1, 1)
            zero_point = zero_point.view(1, 1)

        # Quantize: round((x / scale) + zero_point)
        quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)

        # Dequantize: scale * (quantized - zero_point)
        dequantized = scale * (quantized - zero_point)

        return dequantized

    def to(self, dtype):
        return self.quantize_dequantize(
            self.tensor, self.scale, self.zero_point, self.n_bits, self.layerwise
        ).to(dtype)


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        use_low_rank=False,
        low_rank_dim=None,
        low_rank_alpha=1.0,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        self.use_low_rank = use_low_rank
        self.low_rank_alpha = low_rank_alpha
        self.low_rank_dim = low_rank_dim

        # Low rank branch
        if self.use_low_rank:
            assert low_rank_dim is not None, "low_rank_dim must be specified when use_low_rank=True"
            in_features = self.weight.shape[1]
            out_features = self.weight.shape[0]
            self.low_rank_A = nn.Parameter(torch.randn(out_features, low_rank_dim) * 0.01)
            self.low_rank_B = nn.Parameter(torch.randn(low_rank_dim, in_features) * 0.01)

        # Params for weight quant
        if self.w_bits < 16:
            if self.weight_layerwise:
                self.weight_scale = nn.Parameter(torch.empty(1, 1))
                self.weight_zero_point = nn.Parameter(torch.empty(1, 1))
            else:
                self.weight_scale = nn.Parameter(torch.empty(self.weight.shape[0], 1))
                self.weight_zero_point = nn.Parameter(torch.empty(self.weight.shape[0], 1))

    def forward(self, input_):
        final_weight = self.effective_weight(dtype=input_.dtype)
        out = nn.functional.linear(input_, final_weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"w_bits={self.w_bits}, weight_layerwise={self.weight_layerwise}, "
            f"use_low_rank={self.use_low_rank}, low_rank_dim={self.low_rank_dim}, "
            f"low_rank_alpha={self.low_rank_alpha}"
        )

    # -----------------------------
    # Quant init / helpers
    # -----------------------------
    @torch.no_grad()
    def _compute_qparams_from_tensor(self, calib_W: torch.Tensor, eps: float = 1e-8):
        """
        주어진 텐서(calib_W)를 기준으로 min-max affine 양자화 파라미터를 계산해 반환.
        self.weight_layerwise에 따라 layerwise / per-channel 수행.
        반환값: (scale, zero_point)  # 둘 다 calib_W와 같은 device/dtype로
        """
        if self.w_bits >= 16:
            return None, None  # 양자화 미사용

        qmin, qmax = 0.0, float(2 ** self.w_bits - 1)
        device, dtype = calib_W.device, calib_W.dtype

        if self.weight_layerwise:
            w_min = torch.min(calib_W)
            w_max = torch.max(calib_W)
            w_range = torch.clamp(w_max - w_min, min=eps)
            scale = (w_range / (qmax - qmin)).to(dtype).to(device)
            zero_point = torch.clamp(torch.round(qmin - w_min / scale), qmin, qmax).to(dtype).to(device)
            return scale.view(1, 1), zero_point.view(1, 1)
        else:
            # out-channel(행)별 min-max
            w_min = torch.min(calib_W, dim=1, keepdim=True).values
            w_max = torch.max(calib_W, dim=1, keepdim=True).values
            w_range = torch.clamp(w_max - w_min, min=eps)
            scale = (w_range / (qmax - qmin)).to(dtype).to(device)
            zero_point = torch.clamp(torch.round(qmin - w_min / scale), qmin, qmax).to(dtype).to(device)
            return scale, zero_point

    @torch.no_grad()
    def initialize_quant_params(self, eps: float = 1e-8):
        """
        (LoftQ 호환) Min-Max affine 초기화.
        - use_low_rank=True, w_bits<=8 => (W - α·A@B) 기준으로 파라미터 설정
        - 그 외 => W 기준
        """
        if self.w_bits >= 16:
            return  # 양자화 안 함

        if self.use_low_rank and self.w_bits <= 8:
            LR = torch.matmul(self.low_rank_A, self.low_rank_B) * self.low_rank_alpha
            calib_W = self.weight - LR
        else:
            calib_W = self.weight

        scale, zero_point = self._compute_qparams_from_tensor(calib_W, eps=eps)
        # 파라미터 텐서에 반영
        self.weight_scale.data.copy_(scale)
        self.weight_zero_point.data.copy_(zero_point)

    @torch.no_grad()
    def svd_init_low_rank(self, reduced_rank: int = None, eps: float = 1e-8):
        """
        [LoftQ 스타일 SVD 초기화(단일 스텝)]
        Δ(=αAB)=0에서 시작해:
          1) W 기준으로 양자화 파라미터 설정
          2) base = Q(W)  # dequant 포함
          3) E = W - base
          4) E의 랭크-r SVD로 A,B 초기화
        """
        if not self.use_low_rank:
            raise RuntimeError("use_low_rank=False; low-rank branch not enabled.")
        rank = reduced_rank if reduced_rank is not None else self.low_rank_dim

        # w_bits>=16이면 그냥 W의 랭크-r SVD로
        if self.w_bits >= 16:
            L, R = low_rank_decomposition(self.weight, reduced_rank=rank)
            self.low_rank_A.copy_(L)
            self.low_rank_B.copy_(R)
            return

        # 1) qparams: Δ=0 가정
        scale, zero_point = self._compute_qparams_from_tensor(self.weight, eps=eps)
        self.weight_scale.data.copy_(scale)
        self.weight_zero_point.data.copy_(zero_point)

        # 2) base = Q(W)
        base = LinearQuant(self.weight, self.weight_scale, self.weight_zero_point,
                           self.w_bits, self.weight_layerwise).to(self.weight.dtype)

        # 3) E = W - base
        E = (self.weight - base).to(self.weight.dtype)

        # 4) rank-r SVD(E)
        L, R = low_rank_decomposition(E, reduced_rank=rank)
        if L.shape != self.low_rank_A.shape or R.shape != self.low_rank_B.shape:
            raise RuntimeError(
                f"SVD shapes {L.shape},{R.shape} do not match A,B shapes {self.low_rank_A.shape},{self.low_rank_B.shape}"
            )
        self.low_rank_A.copy_(L)
        self.low_rank_B.copy_(R)

    @torch.no_grad()
    def loftq_init(self, iters: int = 10, reduced_rank: int = None, eps: float = 1e-8):
        """
        LoftQ 교대 최적화 초기화:
          반복 t=1..K:
            1) calib_W_t = W - α A_t B_t
            2) (calib_W_t)로 qparams 재추정
            3) base_t = Q(calib_W_t)
            4) E_t = W - base_t
            5) (rank-r) SVD(E_t) -> A_{t+1}, B_{t+1}
        최종적으로 qparams는 마지막 calib_W_K 기준으로, A,B는 E_K의 랭크-r 근사로 설정.
        """
        if not self.use_low_rank:
            raise RuntimeError("use_low_rank=False; low-rank branch not enabled.")
        rank = reduced_rank if reduced_rank is not None else self.low_rank_dim

        # w_bits>=16이면 양자화가 영향 없으므로 SVD(W)만 수행
        if self.w_bits >= 16:
            L, R = low_rank_decomposition(self.weight, reduced_rank=rank)
            self.low_rank_A.copy_(L)
            self.low_rank_B.copy_(R)
            return

        # 초기 Δ=αAB를 0에서 시작하려면 A,B를 0으로 세팅
        self.low_rank_A.zero_()
        self.low_rank_B.zero_()

        for _ in range(max(1, int(iters))):
            # 1) 현재 Δ로 보정한 대상
            LR = torch.matmul(self.low_rank_A, self.low_rank_B) * self.low_rank_alpha
            calib_W = (self.weight - LR).to(self.weight.dtype)

            # 2) qparams 재계산
            scale, zero_point = self._compute_qparams_from_tensor(calib_W, eps=eps)
            self.weight_scale.data.copy_(scale)
            self.weight_zero_point.data.copy_(zero_point)

            # 3) base = Q(calib_W)
            base = LinearQuant(calib_W, self.weight_scale, self.weight_zero_point,
                               self.w_bits, self.weight_layerwise).to(self.weight.dtype)

            # 4) E = W - base
            E = (self.weight - base).to(self.weight.dtype)

            # 5) rank-r SVD(E) -> A,B 갱신
            L, R = low_rank_decomposition(E, reduced_rank=rank)
            self.low_rank_A.copy_(L)
            self.low_rank_B.copy_(R)

        # 마지막 반복에서의 qparams가 이미 self.*에 들어가 있음
        # 최종 효과 가중치: effective_weight()가 Q(W-αAB)+αAB로 복원

    # -----------------------------
    # NEW: effective weight + error measurement
    # -----------------------------
    @torch.no_grad()
    def effective_weight(self, dtype=None):
        """
        현재 설정으로 추론에 쓰일 최종 가중치 W_eff를 복원합니다.
        - w_bits>=16: (use_low_rank ? W + αAB : W)
        - w_bits<=8:  W_eff = Q(W - αAB) + αAB  (use_low_rank=False면 Q(W))
        """
        W = self.weight
        dtype = W.dtype if dtype is None else dtype

        if self.w_bits >= 16:
            if self.use_low_rank:
                LR = torch.matmul(self.low_rank_A, self.low_rank_B)
                return (W + self.low_rank_alpha * LR).to(dtype)
            return W.to(dtype)

        # <=8bit
        if self.use_low_rank:
            LR = torch.matmul(self.low_rank_A, self.low_rank_B)
            W_for_quant = W - self.low_rank_alpha * LR
        else:
            LR = None
            W_for_quant = W

        base_weight = LinearQuant(
            W_for_quant,
            self.weight_scale,
            self.weight_zero_point,
            self.w_bits,
            self.weight_layerwise,
        ).to(dtype)

        return base_weight if LR is None else (base_weight + self.low_rank_alpha * LR).to(dtype)

    @torch.no_grad()
    def quantization_error(
        self,
        input_: torch.Tensor = None,
        per_channel: bool = False,
        calibrate: bool = False,
        dtype=None,
        eps: float = 1e-12,
    ):
        """
        양자화 오차를 측정합니다.
        - 가중치 오차: E_w = W_eff - W  (subtract-before-quant에서는 순수 양자화 잔차)
        - (옵션) 출력 오차: E_y = Y_q - Y_fp, Y_fp = input @ W^T (+bias)
        Args:
            input_: (N, in_features) 제공 시 출력 오차도 계산
            per_channel: True면 채널별(행별) MSE 포함
            calibrate: True면 측정 전에 min-max로 스케일 재산정(initialize_quant_params)
            dtype: 효과 가중치/출력 계산 dtype (기본은 weight.dtype)
        Returns:
            dict:
              {
                "weight": {
                    "mse","mae","max_abs","rel_fro","snr_db",
                    "per_channel_mse": (optional, [out_features])
                  },
                "output": { ... }  # input_ 제공 시
              }
        """
        if self.w_bits < 16:
            # 스케일이 비어있거나 요청 시 재보정
            need_init = (not hasattr(self, "weight_scale")) or (self.weight_scale.numel() == 0)
            if calibrate or need_init:
                self.initialize_quant_params()

        W = self.weight
        W_eff = self.effective_weight(dtype=dtype)

        # --- weight-domain error
        Ew = (W_eff - W).to(W.dtype)
        mse_w = torch.mean(Ew.pow(2)).item()
        mae_w = torch.mean(Ew.abs()).item()
        max_w = torch.max(Ew.abs()).item()

        fro_W = torch.linalg.norm(W, ord="fro").item()
        fro_E = torch.linalg.norm(Ew, ord="fro").item()
        rel_fro = float(fro_E / (fro_W + eps))
        snr_db = float(20.0 * torch.log10(torch.tensor((fro_W + eps) / (fro_E + eps))).item())

        result = {
            "weight": {
                "mse": mse_w,
                "mae": mae_w,
                "max_abs": max_w,
                "rel_fro": rel_fro,
                "snr_db": snr_db,
            }
        }

        if per_channel:
            # 행별 평균 제곱 오차
            # Ew: (out_features, in_features)
            pcmse = torch.mean(Ew.pow(2), dim=1)  # [out_features]
            result["weight"]["per_channel_mse"] = pcmse.detach().cpu().tolist()

        # --- output-domain error (optional)
        if input_ is not None:
            inp = input_.to(W_eff.dtype)
            y_q = nn.functional.linear(inp, W_eff, self.bias)
            y_fp = nn.functional.linear(inp, W, self.bias)
            Ey = (y_q - y_fp).to(y_fp.dtype)
            mse_y = torch.mean(Ey.pow(2)).item()
            mae_y = torch.mean(Ey.abs()).item()
            max_y = torch.max(Ey.abs()).item()

            # 상대 Frobenius (출력 배치 기준)
            fro_Y = torch.linalg.norm(y_fp, ord="fro").item()
            fro_Ey = torch.linalg.norm(Ey, ord="fro").item()
            rel_fro_y = float(fro_Ey / (fro_Y + eps))
            snr_db_y = float(20.0 * torch.log10(torch.tensor((fro_Y + eps) / (fro_Ey + eps))).item())

            result["output"] = {
                "mse": mse_y,
                "mae": mae_y,
                "max_abs": max_y,
                "rel_fro": rel_fro_y,
                "snr_db": snr_db_y,
            }

        return result

if __name__ == "__main__":
    torch.manual_seed(0)

    # -----------------------------
    # 하이퍼파라미터
    # -----------------------------
    in_features = 500
    out_features = 1000
    w_bits = 4
    weight_layerwise = False
    low_rank_dim = 32
    low_rank_alpha = 1.0
    batch_size = 8
    loftq_iters = 1  # LoftQ 교대 초기화 스텝 수

    # -----------------------------
    # Low-rank 브랜치 사용 모델
    # -----------------------------
    layer = QuantizeLinear(
        in_features, out_features,
        w_bits=w_bits,
        weight_layerwise=weight_layerwise,
        use_low_rank=True,
        low_rank_dim=low_rank_dim,
        low_rank_alpha=low_rank_alpha,
    )
    print(f"start LoftQ initialize (iters={loftq_iters})")
    # LoftQ: Q(W - αAB)와 A,B를 교대로 맞추는 초기화
    layer.loftq_init(iters=loftq_iters)

    # 1) 가중치 기준 오차만
    err_w = layer.quantization_error()
    print(err_w)

    # 2) 입력 배치로 출력 오차까지
    x = torch.randn(batch_size, in_features)
    err = layer.quantization_error(input_=x, per_channel=True)

    # -----------------------------
    # (추가) low-rank 사용 vs 미사용 비교
    # -----------------------------
    layer_nolr = QuantizeLinear(
        in_features, out_features,
        w_bits=w_bits,
        weight_layerwise=weight_layerwise,
        use_low_rank=False,           # 비교 대상: low-rank 미사용
        low_rank_dim=None,
        low_rank_alpha=1.0,
    )
    with torch.no_grad():
        layer_nolr.weight.copy_(layer.weight)
        if layer.bias is not None:
            layer_nolr.bias.copy_(layer.bias)
    breakpoint()

    # 두 설정 모두 같은 방식으로 양자화 파라미터를 재보정한 뒤 오차 측정
    # (low-rank 쪽은 initialize_quant_params가 W-αAB 기준으로 재설정)
    err_lr   = layer.quantization_error(input_=x, per_channel=False, calibrate=True)
    err_nolr = layer_nolr.quantization_error(input_=x, per_channel=False, calibrate=True)

    def _extract(metrics_dict):
        return (
            metrics_dict["weight"]["snr_db"],
            metrics_dict["weight"]["mse"],
            metrics_dict["output"]["snr_db"],
            metrics_dict["output"]["mse"],
        )

    wsnr_lr, wmse_lr, ysnr_lr, ymse_lr   = _extract(err_lr)
    wsnr_nl, wmse_nl, ysnr_nl, ymse_nl   = _extract(err_nolr)

    print("\n=== Low-rank 사용 여부 비교 (w_bits = {}, r = {}, alpha = {}) ===".format(
        w_bits, low_rank_dim, low_rank_alpha
    ))
    print(f"{'':12} | {'Weight SNR(dB)':>14} | {'Weight MSE':>12} | {'Output SNR(dB)':>15} | {'Output MSE':>12}")
    print("-" * 80)
    print(f"{'low-rank':12} | {wsnr_lr:14.3f} | {wmse_lr:12.6e} | {ysnr_lr:15.3f} | {ymse_lr:12.6e}")
    print(f"{'no low-rank':12} | {wsnr_nl:14.3f} | {wmse_nl:12.6e} | {ysnr_nl:15.3f} | {ymse_nl:12.6e}")

    # 간단 판정: 출력 SNR이 큰 쪽, 동률이면 출력 MSE가 작은 쪽
    better = "low-rank" if (ysnr_lr > ysnr_nl or (ysnr_lr == ysnr_nl and ymse_lr < ymse_nl)) else "no low-rank"
    print(f"-> Better (by output SNR/MSE): {better}")
