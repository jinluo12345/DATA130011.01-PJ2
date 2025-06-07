import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math

def plot_epoch_metrics(experiment_dirs, keys_to_plot, save_path=None, style_name="seaborn-darkgrid"):
    """
    Plots epoch-wise metrics for multiple experiment folders，并在 legend 中标明“最佳值”。
    如果 keys_to_plot 长度为 3，则使用 1×3 子图布局，否则沿用“最接近正方形”的自动布局。

    Args:
        experiment_dirs (list of str): List of paths to experiment folders (training_data directories).
                                       每个文件夹须包含：
                                         - "epoch_metrics.csv"
                                         - 唯一的 .yaml 配置文件（如 "bs32.yaml"）
        keys_to_plot (list of str): List of column names in epoch_metrics.csv 要绘制的指标
                                    （如 ['train_loss', 'train_acc', 'val_loss', 'val_acc']）。
        save_path (str or None): 如果提供路径，就把绘制好的整张图保存到该文件；否则只显示不保存。
        style_name (str): Matplotlib 样式名称（如 "seaborn-darkgrid"、"ggplot" 等）。
    """

    # 1) 尝试加载样式；如果找不到，则警告并使用默认样式
    try:
        plt.style.use(style_name)
    except OSError:
        print(f"警告：找不到样式 '{style_name}'，将使用默认样式。")

    num_keys = len(keys_to_plot)
    # --------- 修改：如果恰好有 3 个要画的 key，强制用 1×3 布局 ---------
    if num_keys == 3:
        nrows, ncols = 1, 3
    else:
        # 否则沿用“尽量接近正方形”的自动计算
        ncols = math.ceil(math.sqrt(num_keys))
        nrows = math.ceil(num_keys / ncols)
    # -----------------------------------------------------------------------

    # 2) 创建 Figure 和 Axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    if isinstance(axes, plt.Axes):
        axes = [axes]  # 单子图时转为列表
    else:
        axes = axes.flatten()  # 多子图时展平成一维数组

    # 3) 读取每个实验的数据，并提取 config 名称
    experiment_data = []
    for folder in experiment_dirs:
        csv_path = os.path.join(folder, "epoch_metrics.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"No 'epoch_metrics.csv' found in {folder}")
        df = pd.read_csv(csv_path)

        yaml_files = glob.glob(os.path.join(folder, "*.yaml"))
        if len(yaml_files) != 1:
            print(f"⚠️ 目录 {folder} 下的 yaml 文件不是恰好一个，而是 {len(yaml_files)} 个，跳过。")
            continue
        config_name = os.path.splitext(os.path.basename(yaml_files[0]))[0]
        print(f"✅ 加载到实验：{folder}（config = {config_name}）")
        print(f">>> [{config_name}] epoch_metrics.csv shape: {df.shape}, columns: {df.columns.tolist()}")
        experiment_data.append((config_name, df))

    # 4) 逐个 key 作图，并在 legend 中标明“最佳值”
    for idx, key in enumerate(keys_to_plot):
        ax = axes[idx]
        for config_name, df in experiment_data:
            if key not in df.columns:
                raise KeyError(f"Column '{key}' not found in epoch_metrics.csv of experiment '{config_name}'")

            series = df[key]
            # 判断是 loss 还是 acc，分别取 min 或 max
            if "loss" in key.lower():
                best_val = series.min()
            elif "acc" in key.lower():
                best_val = series.max()
            else:
                # 如果有其他类型指标，可自行扩展逻辑
                best_val = series.iloc[0]

            label = f"{config_name} (best: {best_val:.4f})"
            ax.plot(df["epoch"], series, marker="o", label=label)

        ax.set_title(key.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(key.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True)

    # 5) 隐藏多余的空子图（如果有）
    for j in range(num_keys, nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout()

    # 6) 如果提供了 save_path，就保存到文件
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"✓ 图像已保存到：{save_path}")

    # 7) 最后显示图像
    plt.show()


# ============================
# 示例用法（放在脚本最下方）

if __name__ == "__main__":
    experiments = [
        "/remote-home1/lzjjin/project/fudan-course/DATA130011.01/PJ2/codes/reports/training_data/base",
        "/remote-home1/lzjjin/project/fudan-course/DATA130011.01/PJ2/codes/reports/training_data/ablation-bn/bn",

    ]

    # 假设这次只画 3 个 key
    keys = ["train_loss", "train_acc","val_loss" ,"val_acc"]
    out_file = "/remote-home1/lzjjin/project/fudan-course/DATA130011.01/PJ2/codes/reports/plots/ablation_bn.png"
    plot_epoch_metrics(experiments, keys, save_path=out_file, style_name="seaborn-darkgrid")
