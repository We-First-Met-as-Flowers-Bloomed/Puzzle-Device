# Vision tests

`test_calibration_points.py` 和多数合成场景测试可直接运行。部分 `test_vision.py` 回归用例引用比赛期间采集的实拍图片，以及设备本机生成的 `config.json`；这些文件没有包含在当前开源包中，因此完整运行时会显示缺文件错误。

可直接运行的基础测试：

```powershell
python -m unittest tests.test_calibration_points -v
```

运行完整测试集：

```powershell
python -m unittest tests.test_vision -v
```

若要恢复全部实拍回归测试，请把对应图片放入 `tests/data/`，并由 `config.example.json` 生成经过本机标定的 `config.json`。不要提交包含真实机构标定参数的本机配置。
