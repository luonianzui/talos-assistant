# 第三方组件与规则参考

本工具独立实现规则与界面。第三方规则参考不是官方完整规则保证。

## 软件组件

- PySide6 / Qt for Python：LGPLv3 / GPLv3 / 商业许可，动态库随包分发。https://doc.qt.io/qtforpython-6/licenses.html
- OpenCV：Apache-2.0。https://github.com/opencv/opencv
- RapidOCR：Apache-2.0。https://github.com/RapidAI/RapidOCR
- ONNX Runtime：MIT。https://github.com/microsoft/onnxruntime
- MSS：MIT。https://github.com/BoboTiG/python-mss
- NumPy：BSD-3-Clause。https://github.com/numpy/numpy
- PyInstaller：GPL with bootloader exception。https://pyinstaller.org/en/stable/license.html

打包时通过 `tools/collect_licenses.py` 收集已安装分发包的许可证和元数据。Qt 动态库可替换，应用没有修改 Qt/PySide6 源码；对应版本源代码由上游项目提供。源代码链接和版本随运行包附带。

## OCR 模型

使用 RapidAI 官方模型登记表中固定版本的 PP-OCRv4 中文移动模型。模型由 PaddleOCR 训练、RapidOCR 发布 ONNX 版本；遵循各上游模型许可。字典来自 PaddleOCR release/2.7。下载 URL、模型 SHA256 和版本记录在 `models/provenance.json`。

- https://github.com/RapidAI/RapidOCR
- https://github.com/PaddlePaddle/PaddleOCR
- https://www.modelscope.cn/models/RapidAI/RapidOCR

## 规则依据

- 官方 SS14 公告：https://torchlight.xd.com/ss14pn?lang=zh-CN
- TLIDB 术语：https://tlidb.com/cn/Tip （站点注明测试数据）
- 中文练习模拟器：https://torchlightskill.com/talos/ （非官方，部分参数为练习设定）
- 社区行为报告：https://github.com/blahdot3h/tartarus-simulator （仅参考公开行为描述，未复制代码或素材）
