# 2024 冬季 CUDA 作业

参照 [ONNX 的 Gather 算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)实现一个 CUDA kernel。该 kernel 需要接收三个输入参数：`data`、`indices` 和 `axis`，并输出 `output`。`axis` 的值缺省为 0。在开始实现之前，建议先访问 cuDNN 的官方网站，查询是否有现成的库函数可用于实现 Gather 操作；如果存在现成的库函数，可以同时添加库函数以及手写 cuda 算子的实现作为对比。

1. fork 本仓库；
2. 在 `src/gather/gpu` 目录下添加名为 `gather_cuda.cu` 的文件，并在其中实现 Gather 算子；
3. Gather 需要支持 ONNX 官方文档中 `data` 的所有数据类型，并与 PyTorch 保持 0 误差；
4. 在框架的 `test/gather.py`，验证你编写的 Gather 算子的精度和性能，必要的规模包括：

    - `(3, 2), (2, 2), 0`
      > 表示 `data` 的形状为 `(3, 2)`，`indices` 的形状为 `(2, 2)`，且 `axis` 的值为 0；
    - `(3, 2), (1, 2), 1`；
    - `(50257, 768), (16, 1024), 0`；

5. 性能要求：与 PyTorch 的实现相比，你编写的 Gather 算子在 `(50257, 768), (16, 1024), 0` 的时间开销最多慢 10%；
6. 完成后向原仓库提交一个 issue，标题为`提交作业`，内容包括你的仓库链接，以及可展示你测试的环境、规模和性能差距的文本或图片；

## 命令

- 编译项目

  ```shell
  make
  ```

- 清理项目

  ```shell
  make clean
  ```

- 测试算子

  ```shell
  make test
  ```

## 情况说明

根基实测，天数上可以达到下图所示的性能。由于天数服务器使用虚拟硬件，计时不稳定，需要多试几次才能找到一个 make sense 的数据。

```plaintext
Timestamp    Wed Jan 15 15:31:31 2025
+-----------------------------------------------------------------------------+
|  IX-ML: 3.2.1       Driver Version: 3.2.1       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V100         | 00000000:0C:00.0     | 1500MHz   1200MHz    |
| 0%   33C   P0    58W / 250W   | 0MiB / 16384MiB      | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

![img_v3_02ii_da67acf7-06c5-4e81-aab1-733ff0f1025g](https://github.com/user-attachments/assets/cbb67baf-cae7-4a33-910c-070307f6a268)
