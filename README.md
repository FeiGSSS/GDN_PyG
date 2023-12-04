# 基于超图的多变量异常检测

# 1. 数据收集和处理

### SWAT和WADI数据

处理结果

| 数据 | #节点 | 训练时间步 | 测试时间步 |
| :--: | :---: | :--------: | :--------: |
| SWAT |  51  |   47520   |   44991   |
| WADI |  127  |   118800   |   17280   |

和baseline文章中的数据有分歧，训练和测试时间步都少5步！TODO

生成方式：

```
python ./dataset/swat_preprocesee.py
python ./dataset/wadi_preprocesee.py
```
