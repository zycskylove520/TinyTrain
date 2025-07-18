#!/usr/bin/env python3
# test_progress.py
import time
import random
import threading
from TinyTrain.utils.TT_progress_bar import TTProgressBar


# ---------- 1 ~ 12 号测试函数 ----------
def test_01_simple_for():
    print("=== 1. 最简 for 迭代 ===")
    for _ in TTProgressBar(range(30), desc="for默认"):
        time.sleep(0.05)


def test_02_disable():
    print("\n=== 2. 完全静默 ===")
    with TTProgressBar(range(20), desc="静默", disable=True) as bar:
        for _ in bar:
            time.sleep(0.05)


def test_03_disable_color():
    print("\n=== 3. 禁用颜色 ===")
    with TTProgressBar(range(20), desc="无色", disable_color=True) as bar:
        for _ in bar:
            time.sleep(0.05)


def test_04_with_manual_update():
    print("\n=== 4. with + 手动 update ===")
    with TTProgressBar(total=100, desc="手动") as bar:
        for _ in range(100):
            time.sleep(0.02)
            bar.update()


def test_05_update_bulk():
    print("\n=== 5. with + update_bulk ===")
    with TTProgressBar(total=100, desc="批量") as bar:
        for _ in range(10):
            time.sleep(0.1)
            bar.update_bulk(10)


def test_06_random_color():
    print("\n=== 6. 随机颜色条 ===")
    with TTProgressBar(range(30), desc="彩虹", random_color=True) as bar:
        for _ in bar:
            time.sleep(0.05)


def test_07_show_percent():
    print("\n=== 7. 显示百分比 ===")
    with TTProgressBar(range(30), desc="百分比", show_percent=True) as bar:
        for _ in bar:
            time.sleep(0.05)


def test_08_custom_title():
    print("\n=== 8. 自定义标题 ===")
    with TTProgressBar(range(20), title="【自定义标题】", desc="任务") as bar:
        for _ in bar:
            time.sleep(0.05)


def test_09_dynamic_desc():
    print("\n=== 9. 动态描述 ===")
    with TTProgressBar(range(30), desc="初始") as bar:
        for i in bar:
            if i == 10:
                bar.set_description("已过半")
            if i == 20:
                bar.set_description("即将完成")
            time.sleep(0.05)


def test_10_multithread_nested():
    print("\n=== 10. 多线程嵌套 ===")

    def worker(idx):
        for _ in TTProgressBar(range(15), desc=f"线程{idx}"):
            time.sleep(random.uniform(0.02, 0.08))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_11_set_description():
    print("\n=== 11. set_description 动态描述 ===")
    with TTProgressBar(total=100, desc="初始描述") as bar:
        for i in range(1, 101):
            bar.update()
            if i == 25:
                bar.set_description("加载中 25%")
            elif i == 50:
                bar.set_description("加载中 50%")
            elif i == 75:
                bar.set_description("加载中 75%")
            elif i == 90:
                bar.set_description("即将完成")
            time.sleep(0.02)


def test_12_set_title():
    print("\n=== 12. set_title 动态标题 ===")
    with TTProgressBar(total=60, desc="任务") as bar:
        bar.set_title("【阶段 1】初始化")
        for _ in range(30):
            bar.update()
            time.sleep(0.05)
        bar.set_title("【阶段 2】处理中")
        for _ in range(30):
            bar.update()
            time.sleep(0.05)
        bar.set_title("【阶段 3】完成")


# ---------- 统一入口 ----------
if __name__ == "__main__":
    tests = [
        test_01_simple_for,
        test_02_disable,
        test_03_disable_color,
        test_04_with_manual_update,
        test_05_update_bulk,
        test_06_random_color,
        test_07_show_percent,
        test_08_custom_title,
        test_09_dynamic_desc,
        test_10_multithread_nested,
        test_11_set_description,
        test_12_set_title,
    ]
    for t in tests:
        t()
    print("\n✅ 所有测试完成！")