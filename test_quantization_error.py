import torch
from models.transformer_flux import (
    FluxTransformer2DModel as FluxTransformer2DModelQuant,
)
from models.utils_quant import QuantizeLinear, LinearQuant
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc


def calculate_layer_errors_gpu(original_weight, quantized_weight, low_rank_weight=None):
    """GPU에서 레이어별 quantization error 계산"""
    with torch.no_grad():
        # Ensure tensors are on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_weight = original_weight.to(device)
        quantized_weight = quantized_weight.to(device)

        # Basic quantization error (GPU에서 계산)
        quant_error = original_weight - quantized_weight
        quant_mse = torch.mean(quant_error**2)
        quant_mae = torch.mean(torch.abs(quant_error))

        # Relative error (GPU에서 계산)
        weight_norm = torch.norm(original_weight)
        quant_relative_error = torch.norm(quant_error) / (weight_norm + 1e-8)

        errors = {
            "quantization_mse": quant_mse.item(),
            "quantization_mae": quant_mae.item(),
            "quantization_relative_error": quant_relative_error.item(),
            "original_weight_norm": weight_norm.item(),
        }

        # Low-rank compensation error if available (GPU에서 계산)
        if low_rank_weight is not None:
            low_rank_weight = low_rank_weight.to(device)
            compensated_weight = quantized_weight + low_rank_weight
            compensation_error = original_weight - compensated_weight
            comp_mse = torch.mean(compensation_error**2)
            comp_mae = torch.mean(torch.abs(compensation_error))
            comp_relative_error = torch.norm(compensation_error) / (weight_norm + 1e-8)

            # Error reduction metrics (GPU에서 계산)
            mse_reduction = (quant_mse - comp_mse) / (quant_mse + 1e-8)
            mae_reduction = (quant_mae - comp_mae) / (quant_mae + 1e-8)
            relative_error_reduction = (quant_relative_error - comp_relative_error) / (
                quant_relative_error + 1e-8
            )

            errors.update(
                {
                    "compensation_mse": comp_mse.item(),
                    "compensation_mae": comp_mae.item(),
                    "compensation_relative_error": comp_relative_error.item(),
                    "mse_reduction_ratio": mse_reduction.item(),
                    "mae_reduction_ratio": mae_reduction.item(),
                    "relative_error_reduction_ratio": relative_error_reduction.item(),
                }
            )

        return errors


def extract_weights_batch_gpu(model, w_bits, batch_size=8):
    """GPU에서 배치 단위로 weights 추출 및 에러 계산"""
    layer_errors = {}
    layer_data = []

    # 모든 레이어 정보를 먼저 수집
    for name, module in model.named_modules():
        if isinstance(module, QuantizeLinear):
            original_weight = module.weight.detach().clone()

            # Calculate quantized weight on GPU
            if w_bits < 16:
                quantized_weight = LinearQuant(
                    original_weight,
                    module.weight_scale,
                    module.weight_zero_point,
                    w_bits,
                    layerwise=False,
                ).to(original_weight.dtype)
            else:
                quantized_weight = original_weight

            # Get low-rank weight if available
            low_rank_weight = None
            if hasattr(module, "low_rank_A") and hasattr(module, "low_rank_B"):
                low_rank_weight = (
                    torch.matmul(module.low_rank_A, module.low_rank_B)
                    * module.low_rank_alpha
                )

            layer_data.append(
                (name, original_weight, quantized_weight, low_rank_weight)
            )

    for i in tqdm(
        range(0, len(layer_data), batch_size), desc="Processing layers on GPU"
    ):
        batch = layer_data[i : i + batch_size]

        # 배치를 GPU로 이동하고 병렬 계산
        batch_errors = []

        for name, original_weight, quantized_weight, low_rank_weight in batch:
            # GPU에서 에러 계산
            errors = calculate_layer_errors_gpu(
                original_weight, quantized_weight, low_rank_weight
            )
            batch_errors.append((name, errors))

        # 결과 저장
        for name, errors in batch_errors:
            layer_errors[name] = errors

        # GPU 메모리 정리
        torch.cuda.empty_cache()

    return layer_errors


def calculate_quantized_weights_gpu_batch(model, w_bits):
    """GPU에서 배치로 quantized weights 계산"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    quantized_weights = {}
    named_params = dict(model.named_parameters())

    # weight parameters를 배치로 처리
    weight_params = [
        (name, param)
        for name, param in named_params.items()
        if "weight" in name
        and not any(x in name for x in ["scale", "zero_point", "low_rank"])
    ]

    for name, weight_param in tqdm(
        weight_params, desc="Calculating quantized weights on GPU"
    ):
        scale_name = name.replace("weight", "weight_scale")
        zero_point_name = name.replace("weight", "weight_zero_point")

        if scale_name in named_params and zero_point_name in named_params:
            weight_scale = named_params[scale_name]
            weight_zero_point = named_params[zero_point_name]

            # GPU에서 quantization 계산
            quantized_weight = LinearQuant(
                weight_param.to(device),
                weight_scale.to(device),
                weight_zero_point.to(device),
                w_bits,
                layerwise=False,
            ).to(weight_param.dtype)

            quantized_weights[name] = quantized_weight

    return quantized_weights


def load_and_analyze_model_task_gpu(args):
    """GPU 활용한 단일 모델 설정 분석"""
    model_name, w_bits, low_rank_dim = args

    try:
        print(f"Processing {w_bits}-bit, rank-{low_rank_dim} on GPU")

        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        # Load model on GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FluxTransformer2DModelQuant.from_pretrained(
            pretrained_model_name_or_path=model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            device_map=None,
            w_bits=w_bits,
            use_low_rank=low_rank_dim > 0,
            low_rank_dim=low_rank_dim if low_rank_dim > 0 else None,
            low_rank_alpha=1.0,
        ).to(device)

        if torch.cuda.is_available():
            print(
                f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        # Initialize quantization parameters
        if w_bits <= 8:
            from generate_images import calculate_scale_and_zero_point

            weight_scale_dict = {}
            scale_tasks = []
            named_params = dict(model.named_parameters())

            # GPU에서 스케일 계산을 위한 태스크 준비
            for name, param in named_params.items():
                if "weight_scale" in name:
                    weight_name = name.replace("weight_scale", "weight")
                    weight_param = named_params.get(weight_name, None)
                    if weight_param is not None:
                        scale_tasks.append((name, weight_param.to(device), w_bits))

            # GPU에서 병렬 스케일 계산
            with ThreadPoolExecutor(max_workers=4) as executor:  # GPU 메모리 제한
                scale_results = list(
                    tqdm(
                        executor.map(calculate_scale_and_zero_point, scale_tasks),
                        total=len(scale_tasks),
                        desc=f"Calculating scales on GPU for {w_bits}-bit",
                    )
                )

            for scale_name, scale, zero_point_name, zero_point in scale_results:
                weight_scale_dict[scale_name] = scale.to(device)
                weight_scale_dict[zero_point_name] = zero_point.to(device)

            model.load_state_dict(weight_scale_dict, assign=True, strict=False)

            # GPU에서 low-rank 초기화
            if low_rank_dim > 0:
                from generate_images import initialize_low_rank_with_svd_parallel

                print(f"Initializing low-rank on GPU for rank-{low_rank_dim}")
                initialize_low_rank_with_svd_parallel(model, w_bits, low_rank_dim)

        # GPU에서 배치 단위로 에러 분석
        layer_errors = extract_weights_batch_gpu(
            model, w_bits, batch_size=4
        )  # 메모리에 따라 조정

        if torch.cuda.is_available():
            print(
                f"GPU memory after analysis: {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return f"{w_bits}bit", f"rank{low_rank_dim}", layer_errors

    except Exception as e:
        print(f"Error processing {w_bits}-bit, rank-{low_rank_dim}: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"{w_bits}bit", f"rank{low_rank_dim}", {}


def analyze_quantization_errors_gpu_optimized(
    model_name, w_bits_list=[8, 4, 2], low_rank_dims=[0, 8, 16]
):
    """GPU 최적화된 quantization error 분석"""

    analysis_tasks = []
    for w_bits in w_bits_list:
        for low_rank_dim in low_rank_dims:
            analysis_tasks.append((model_name, w_bits, low_rank_dim))

    print(f"Starting GPU-optimized analysis of {len(analysis_tasks)} configurations...")

    results = {}

    # GPU 메모리 최적화를 위해 한 번에 하나씩 처리
    for task in tqdm(analysis_tasks, desc="Processing configurations on GPU"):
        bit_config, rank_config, layer_errors = load_and_analyze_model_task_gpu(task)

        if bit_config not in results:
            results[bit_config] = {}
        results[bit_config][rank_config] = layer_errors

        # 각 설정 후 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def calculate_statistics_gpu_batch(results):
    """GPU에서 배치로 통계 계산"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}

    for bit_config, rank_configs in results.items():
        summary[bit_config] = {}

        for rank_config, layer_errors in tqdm(
            rank_configs.items(), desc=f"Calculating stats for {bit_config}"
        ):
            if not layer_errors:
                summary[bit_config][rank_config] = {}
                continue

            # GPU에서 텐서로 변환하여 배치 계산
            all_quant_mse = torch.tensor(
                [errors["quantization_mse"] for errors in layer_errors.values()],
                device=device,
            )
            all_quant_mae = torch.tensor(
                [errors["quantization_mae"] for errors in layer_errors.values()],
                device=device,
            )
            all_quant_rel = torch.tensor(
                [
                    errors["quantization_relative_error"]
                    for errors in layer_errors.values()
                ],
                device=device,
            )

            # GPU에서 통계 계산
            stats = {
                "avg_quantization_mse": torch.mean(all_quant_mse).item(),
                "std_quantization_mse": torch.std(all_quant_mse).item(),
                "avg_quantization_mae": torch.mean(all_quant_mae).item(),
                "std_quantization_mae": torch.std(all_quant_mae).item(),
                "avg_quantization_relative_error": torch.mean(all_quant_rel).item(),
                "std_quantization_relative_error": torch.std(all_quant_rel).item(),
                "num_layers": len(layer_errors),
            }

            # Low-rank compensation statistics
            comp_mse_list = [
                errors.get("compensation_mse")
                for errors in layer_errors.values()
                if "compensation_mse" in errors
            ]

            if comp_mse_list and all(x is not None for x in comp_mse_list):
                comp_mse_tensor = torch.tensor(comp_mse_list, device=device)
                mse_reductions = torch.tensor(
                    [
                        errors["mse_reduction_ratio"]
                        for errors in layer_errors.values()
                        if "mse_reduction_ratio" in errors
                    ],
                    device=device,
                )
                mae_reductions = torch.tensor(
                    [
                        errors["mae_reduction_ratio"]
                        for errors in layer_errors.values()
                        if "mae_reduction_ratio" in errors
                    ],
                    device=device,
                )
                rel_reductions = torch.tensor(
                    [
                        errors["relative_error_reduction_ratio"]
                        for errors in layer_errors.values()
                        if "relative_error_reduction_ratio" in errors
                    ],
                    device=device,
                )

                stats.update(
                    {
                        "avg_compensation_mse": torch.mean(comp_mse_tensor).item(),
                        "std_compensation_mse": torch.std(comp_mse_tensor).item(),
                        "avg_mse_reduction_ratio": torch.mean(mse_reductions).item(),
                        "std_mse_reduction_ratio": torch.std(mse_reductions).item(),
                        "avg_mae_reduction_ratio": torch.mean(mae_reductions).item(),
                        "std_mae_reduction_ratio": torch.std(mae_reductions).item(),
                        "avg_relative_error_reduction_ratio": torch.mean(
                            rel_reductions
                        ).item(),
                        "std_relative_error_reduction_ratio": torch.std(
                            rel_reductions
                        ).item(),
                    }
                )

            summary[bit_config][rank_config] = stats

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def create_plot_task(args):
    """개별 플롯 생성을 위한 병렬 처리 함수"""
    plot_type, summary, output_dir = args

    output_dir = Path(output_dir)

    if plot_type == "mse_comparison":
        # MSE comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, (bit_config, rank_data) in enumerate(summary.items()):
            if i >= 4:  # 최대 4개 subplot
                break

            ax = axes[i // 2, i % 2]

            ranks = []
            quant_mse = []
            comp_mse = []

            for rank_config, stats in rank_data.items():
                if not stats:  # 빈 stats 체크
                    continue
                rank_num = int(rank_config.replace("rank", ""))
                ranks.append(rank_num)
                quant_mse.append(stats["avg_quantization_mse"])

                if "avg_compensation_mse" in stats:
                    comp_mse.append(stats["avg_compensation_mse"])
                else:
                    comp_mse.append(stats["avg_quantization_mse"])

            if ranks:  # 데이터가 있을 때만 플롯
                ax.plot(
                    ranks,
                    quant_mse,
                    "o-",
                    label="Quantization Only",
                    color="red",
                    linewidth=2,
                )
                ax.plot(
                    ranks,
                    comp_mse,
                    "s-",
                    label="With Low-rank Compensation",
                    color="blue",
                    linewidth=2,
                )
                ax.set_xlabel("Low-rank Dimension")
                ax.set_ylabel("Average MSE")
                ax.set_title(f"{bit_config} Quantization Error")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "mse_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    elif plot_type == "error_reduction":
        # Error reduction plot
        fig, ax = plt.subplots(figsize=(12, 8))

        for bit_config, rank_data in summary.items():
            ranks = []
            reductions = []

            for rank_config, stats in rank_data.items():
                if not stats:  # 빈 stats 체크
                    continue
                rank_num = int(rank_config.replace("rank", ""))
                if rank_num > 0 and "avg_mse_reduction_ratio" in stats:
                    ranks.append(rank_num)
                    reductions.append(stats["avg_mse_reduction_ratio"] * 100)

            if ranks:
                ax.plot(
                    ranks,
                    reductions,
                    "o-",
                    label=f"{bit_config}",
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Low-rank Dimension")
        ax.set_ylabel("MSE Reduction (%)")
        ax.set_title("Quantization Error Reduction by Low-rank Compensation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_dir / "error_reduction.png", dpi=300, bbox_inches="tight")
        plt.close()

    return f"{plot_type} completed"


def plot_error_analysis_parallel(summary, output_dir="output/error_analysis"):
    """Error 분석 결과 시각화 (병렬화)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 플롯 생성 태스크
    plot_tasks = [
        ("mse_comparison", summary, output_dir),
        ("error_reduction", summary, output_dir),
    ]

    # 병렬로 플롯 생성
    with ThreadPoolExecutor(max_workers=2) as executor:
        plot_results = list(
            tqdm(
                executor.map(create_plot_task, plot_tasks),
                total=len(plot_tasks),
                desc="Creating plots",
            )
        )

    print(f"Plots saved to {output_dir}")
    return plot_results


def save_results_parallel(results, summary, output_dir):
    """결과 저장을 병렬화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_detailed():
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return "detailed_results.json saved"

    def save_summary():
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return "summary.json saved"

    # 병렬로 파일 저장
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(save_detailed), executor.submit(save_summary)]

        for future in as_completed(futures):
            print(future.result())


def main():
    model_name = "black-forest-labs/FLUX.1-dev"

    # GPU 상태 확인
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(
            f"Available GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB"
        )
    else:
        print("GPU not available, using CPU")

    print("Starting GPU-optimized quantization error analysis...")
    results = analyze_quantization_errors_gpu_optimized(
        model_name,
        w_bits_list=[8, 4],  # 시작은 작게
        low_rank_dims=[0, 8, 16],  # 시작은 작게
    )

    print("\nCalculating statistics on GPU...")
    summary = calculate_statistics_gpu_batch(results)

    # Save results in parallel
    output_dir = Path("output/error_analysis")
    print("\nSaving results...")
    save_results_parallel(results, summary, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("QUANTIZATION ERROR ANALYSIS SUMMARY (GPU-Optimized)")
    print("=" * 80)

    for bit_config, rank_data in summary.items():
        print(f"\n{bit_config.upper()} QUANTIZATION:")
        print("-" * 50)

        for rank_config, stats in rank_data.items():
            if not stats:  # 빈 stats 체크
                continue
            rank_num = int(rank_config.replace("rank", ""))
            print(f"  Rank {rank_num}:")
            print(f"    Avg Quantization MSE: {stats['avg_quantization_mse']:.6e}")

            if "avg_compensation_mse" in stats:
                print(f"    Avg Compensation MSE: {stats['avg_compensation_mse']:.6e}")
                print(f"    MSE Reduction: {stats['avg_mse_reduction_ratio']*100:.2f}%")
                print(
                    f"    Relative Error Reduction: {stats['avg_relative_error_reduction_ratio']*100:.2f}%"
                )

    # Create visualizations in parallel
    print("\nGenerating plots in parallel...")
    plot_error_analysis_parallel(summary)

    print(f"\nGPU-optimized analysis complete! Results saved to {output_dir}")

    # 최종 GPU 메모리 상태
    if torch.cuda.is_available():
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
