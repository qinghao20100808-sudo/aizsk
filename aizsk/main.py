"""
main.py - AI 实时受力分析系统入口

用法:
    python -m aizsk.main              # 启动 Web 服务 + 摄像头
    python -m aizsk.main --demo       # 演示模式（推荐，无需摄像头）
    python -m aizsk.main --no-cam     # 无摄像头/无图像模式
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEMO_SCENES = {
    "incline": "斜面滑块（推荐）",
    "flat_push": "水平推动",
    "free_fall": "自由落体",
    "pendulum": "单摆",
}


def main():
    parser = argparse.ArgumentParser(
        description="AI 实时受力分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m aizsk.main                    # 默认启动（摄像头模式）
    python -m aizsk.main --demo             # 演示模式（推荐，无需摄像头）
    python -m aizsk.main --demo incline     # 指定演示场景
    python -m aizsk.main --port 8080        # 指定端口
        """,
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Web 服务器端口 (默认: 5000)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="绑定地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="摄像头 ID (默认: 0)"
    )
    parser.add_argument(
        "--win-cam",
        action="store_true",
        help="Windows 共享文件摄像头模式（WSL 专用）",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="画面宽度 (默认: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="画面高度 (默认: 480)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Web模式 - 仅启动服务器，前端浏览器负责摄像头捕获 (推荐用于部署)",
    )
    parser.add_argument(
        "--no-cam",
        action="store_true",
        help="无摄像头模式（用于调试前端，无画面）",
    )
    parser.add_argument(
        "--demo",
        type=str,
        nargs="?",
        const="incline",
        default=None,
        help=f"演示模式（无需摄像头）可选场景: {', '.join(DEMO_SCENES.keys())}",
    )

    args = parser.parse_args()

    # 打印启动信息
    print("=" * 50)
    print("  ⚡ AI 实时受力分析系统 v0.1.0")
    print("=" * 50)
    print()

    # 演示模式（优先）
    if args.demo is not None:
        from .web_server import SimulationStream, WebServer

        if args.demo not in DEMO_SCENES:
            print(f"  ⚠️  未知场景 '{args.demo}'，可用: {', '.join(DEMO_SCENES.keys())}")
            print(f"  使用默认场景: incline")
            args.demo = "incline"

        logger.info(f"演示模式启动 - 场景: {DEMO_SCENES[args.demo]}")
        sim = SimulationStream(
            scene_id=args.demo,
            width=args.width,
            height=args.height,
        )
        sim.start()

        server = WebServer(host=args.host, port=args.port)
        print(f"  🌐 打开浏览器访问: http://localhost:{args.port}")
        print()
        print(f"  🎯 场景: {DEMO_SCENES[args.demo]}")
        print()
        print("  📋 操作说明:")
        print("     1. 点击画面中的物体选中它")
        print("     2. 系统自动进行受力分析")
        print("     3. 可以在侧栏手动切换场景类型")
        print("     4. 按 R 键重置选择")
        print()
        print("  ⏹  按 Ctrl+C 停止")
        print()

        try:
            server.start(sim)
        except KeyboardInterrupt:
            logger.info("正在停止...")
        finally:
            sim.stop()
            logger.info("已停止")
        return

    # Web 模式 - 浏览器摄像头（推荐）
    if args.web:
        logger.info("🌐 Web 模式启动（浏览器捕获摄像头）")
        from .web_server import WebServer

        server = WebServer(host=args.host, port=args.port)
        print("=" * 50)
        print("  ⚡ AI 实时受力分析系统 v2.0")
        print("  📷 浏览器摄像头模式 (推荐)")
        print("=" * 50)
        print()
        print(f"  🌐 打开浏览器访问: http://localhost:{args.port}")
        print()
        print("  📋 操作说明:")
        print("     1. 点击「启动摄像头」按钮")
        print("     2. 允许浏览器使用摄像头")
        print("     3. 点击画面中的物体进行受力分析")
        print("     4. 在侧栏选择场景类型（可选）")
        print("     5. 按 R 键重置选择")
        print()
        print("  💡 提示: 首次加载需要下载 YOLO 模型 (~5MB)")
        print("  ⏹  按 Ctrl+C 停止")
        print()

        try:
            server.start()
        except KeyboardInterrupt:
            logger.info("正在停止...")
        logger.info("已停止")
        return

    if args.no_cam:
        logger.info("无摄像头模式启动")
        from .web_server import CameraStream, WebServer

        camera = CameraStream()
        camera.model_loaded = False
        server = WebServer(host=args.host, port=args.port)
        print(f"  🌐 打开浏览器访问: http://localhost:{args.port}")
        print()
        print("  💡 提示: 无摄像头模式无法进行检测")
        print()
        server.start(camera)
        return

    # Windows 摄像头模式
    if args.win_cam:
        logger.info("Windows 共享文件摄像头模式启动")
        from .web_server import CameraStream, WebServer

        camera = CameraStream(
            width=args.width,
            height=args.height,
            use_windows_cam=True,
        )

        if not camera.start():
            logger.error("❌ Windows 摄像头启动失败")
            logger.info("请确保 Windows 上已启动 ffmpeg 摄像头采集")
            sys.exit(1)

        server = WebServer(host=args.host, port=args.port)
        print(f"  🌐 打开浏览器访问: http://localhost:{args.port}")
        print(f"  📷 模式: Windows 共享文件摄像头")
        print()
        print("  📋 操作说明:")
        print("     1. 点击画面中的物体选中它")
        print("     2. 系统自动进行 YOLO 检测 + 受力分析")
        print("     3. 按 R 键重置选择")
        print()
        print("  ⏹  按 Ctrl+C 停止")
        print()

        try:
            server.start(camera)
        except KeyboardInterrupt:
            logger.info("正在停止...")
        finally:
            camera.stop()
            logger.info("已停止")
        return

    # 正常模式：摄像头 + YOLO
    logger.info("初始化摄像头...")
    from .web_server import CameraStream, WebServer

    camera = CameraStream(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
    )

    if not camera.start():
        logger.error("❌ 摄像头启动失败")
        logger.info("提示: 试试 --demo 模式进行演示，或 --no-cam 模式调试前端")
        sys.exit(1)

    print()
    print(f"  🌐 打开浏览器访问: http://localhost:{args.port}")
    print()
    print("  📋 操作说明:")
    print("     1. 在浏览器中点击画面中的物体")
    print("     2. 系统自动检测、跟踪并分析受力")
    print("     3. 可以在侧栏手动选择场景类型")
    print("     4. 按 R 键重置选择")
    print()
    print("  ⏹  按 Ctrl+C 停止")
    print()

    try:
        server = WebServer(host=args.host, port=args.port)
        server.start(camera)
    except KeyboardInterrupt:
        logger.info("正在停止...")
    finally:
        camera.stop()
        logger.info("已停止")


if __name__ == "__main__":
    main()
