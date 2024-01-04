# <div align="center"><strong>vLLM</strong></div>
## 简介
vLLM是一个快速且易于使用的LLM推理和服务库，使用PageAttention高效管理kv内存，Continuous batching传入请求，支持很多Hugging Face模型，如LLaMA & LLaMA-2、Qwen、Chatglm2 & Chatglm23等。

## 安装
vLLM支持
+ Python 3.8.
+ Python 3.9.
+ Python 3.10.
+ Python 3.11.

### 使用源码编译方式安装

#### 编译环境准备
提供2种环境准备方式：

1. 基于光源pytorch基础镜像环境：镜像下载地址：[https://sourcefind.cn/#/image/dcu/pytorch](https://sourcefind.cn/#/image/dcu/pytorch)，根据pytorch、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境：安装pytorch，pytorch whl包下载目录：[https://cancon.hpccube.com:65024/4/main/pytorch/dtk23.10](https://cancon.hpccube.com:65024/4/main/pytorch/dtk23.10)，根据python、dtk版本,下载对应pytorch的whl包。安装命令如下：
```shell
pip install torch* (下载的torch的whl包)
pip install setuptools wheel
```

#### 源码编译安装
```shell
git clone https://developer.hpccube.com/codes/aicomponent/vllm # 根据需要的分支进行切换
```

- 提供2种源码编译方式（进入vllm目录）：
```
1. 编译whl包并安装
python setup.py bdist_wheel 
pip install dist/vllm*

2. 源码编译安装
python3 setup.py install 
```

#### 注意事项
+ 若使用 pip install 下载安装过慢，可添加源：-i https://pypi.tuna.tsinghua.edu.cn/simple/

## 验证
- python -c "import vllm; print(vllm.\_\_version__)"，版本号与官方版本同步，查询该软件的版本号，例如0.2.7；

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)