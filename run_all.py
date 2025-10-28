# run_all.py
import os
import sys
import shutil
import subprocess
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(THIS_DIR, "train.py")

# 顺序：先 maddpg，再 madqn，然后 mappo，最后 qmix（你可按需调整顺序）
ALGOS = ["maddpg", "mappo", "madqn", "qmix"]

def run_once(algo: str, device: str, max_episodes: int, test_every: int, show: bool):
    """调用 train.py 训练一个算法"""
    cmd = [
        sys.executable, TRAIN_PY,
        "--algo", algo,
        "--device", device,
        "--max_episodes", str(max_episodes),
        "--test_every_episodes", str(test_every),
        "--model_dir", "model"  # 让每次都写入同一个源目录，之后再备份到 *_<algo>
    ]
    if show:
        cmd.append("--show")

    print(f"\n==================== Run {algo} ====================")
    print("Command:", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"⚠ 训练 {algo} 进程以非零状态退出（ret={ret})，将继续备份并进行下一个算法。")

def backup_dirs(algo: str):
    """
    无论源目录是否存在，都确保创建目标目录并进行覆盖保存：
      - model        -> model_<algo>
      - train_log    -> train_log_<algo>
    """
    src_model = os.path.join(THIS_DIR, "model")
    src_log   = os.path.join(THIS_DIR, "train_log")

    dst_model = os.path.join(THIS_DIR, f"model_{algo}")
    dst_log   = os.path.join(THIS_DIR, f"train_log_{algo}")

    # 目标目录先创建，保证一定存在
    os.makedirs(dst_model, exist_ok=True)
    os.makedirs(dst_log, exist_ok=True)

    # 覆盖复制模型目录
    if os.path.isdir(src_model):
        shutil.copytree(src_model, dst_model, dirs_exist_ok=True)
        print(f"✅ 模型已保存到 {dst_model}")
    else:
        print(f"⚠ 源模型目录 {src_model} 不存在，已保留空的 {dst_model}")

    # 覆盖复制日志目录
    if os.path.isdir(src_log):
        shutil.copytree(src_log, dst_log, dirs_exist_ok=True)
        print(f"✅ 训练日志已保存到 {dst_log}")
    else:
        print(f"⚠ 源日志目录 {src_log} 不存在，已保留空的 {dst_log}")

def clean_sources():
    """
    可选清理：把通用的 model/ 与 train_log/ 清空，避免下个算法在同一源目录基础上叠加。
    如果你更想累计在同一目录再拷贝，请注释掉本函数调用。
    """
    for name in ["model", "train_log"]:
        p = os.path.join(THIS_DIR, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
    print("🧹 已清空源目录 model/ 与 train_log/")

def parse_args():
    ap = argparse.ArgumentParser("Run multiple MARL algorithms sequentially and back up results")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--max_episodes", type=int, default=500000)
    ap.add_argument("--test_every", type=int, default=1000)
    ap.add_argument("--show", action="store_true", default=False)
    ap.add_argument("--no_clean_between", action="store_true", default=False,
                    help="不在算法之间清空 model/ 和 train_log/（默认会清空）")
    return ap.parse_args()

def main():
    args = parse_args()

    for idx, algo in enumerate(ALGOS):
        # 训练前清空源目录，确保每个算法写到干净的 model/ 和 train_log/
        if idx == 0 or not args.no_clean_between:
            clean_sources()

        run_once(
            algo=algo,
            device=args.device,
            max_episodes=args.max_episodes,
            test_every=args.test_every,
            show=args.show
        )
        # 不论训练成功与否，都备份（并创建对应目标目录）
        backup_dirs(algo)

    print("\n✅ 全部算法执行完毕。结果已保存到：")
    for algo in ALGOS:
        print(f"  - model_{algo}")
        print(f"  - train_log_{algo}")

if __name__ == "__main__":
    main()
