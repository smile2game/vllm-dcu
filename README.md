# <div align="center"><strong>vLLM</strong></div>
## 简介
vLLM是一个快速且易于使用的LLM推理和服务库，使用PageAttention高效管理kv内存，Continuous batching传入请求，支持很多Hugging Face模型，如LLaMA & LLaMA-2、Qwen、Chatglm2 & Chatglm3等。

## 暂不支持的官方功能
- **量化推理**：目前支持fp16的推理和gptq推理，awq-int4和mralin的权重量化、kv-cache fp8推理方案暂不支持
- **模块支持**：目前不支持Sliding window attention、 moe kernel和lora模块


## 支持模型结构列表
|     结构     |     模型      | 模型并行 | FP16 |
| :----------: | :----------: | :------: | :--: |
|    LlamaForCausalLM       |    LLaMA          |   Yes    | Yes  |
|    LlamaForCausalLM       |    LLaMA-2        |   Yes    | Yes  |
|    LlamaForCausalLM       |    LLaMA-3        |   Yes    | Yes  |
|    LlamaForCausalLM       |    Codellama      |   Yes    | Yes  |
|    QWenLMHeadModel        |    QWen           |   Yes    | Yes  |
|    Qwen2ForCausalLM       |    QWen1.5        |   Yes    | Yes  |
|    Qwen2ForCausalLM       |    CodeQwen1.5    |   Yes    | Yes  |
|    Qwen2ForCausalLM       |    QWen2          |   Yes    | Yes  |
|    ChatGLMModel           |    chatglm2       |   Yes    | Yes  |
|    ChatGLMModel           |    chatglm3       |   Yes    | Yes  |
|    ChatGLMModel           |    glm-4          |   Yes    | Yes  |
|    BaiChuanForCausalLM    |    Baichuan-7B    |   Yes    | Yes  |
|    BaiChuanForCausalLM    |    Baichuan2-7B   |   Yes    | Yes  |
|    InternLMForCausalLM    |    InternLM       |   Yes    | Yes  |
|    InternLM2ForCausalLM   |    InternLM2      |   Yes    | Yes  |
|    LlamaForCausalLM       |    deepseek       |   Yes    | Yes  |
|    DeepseekV2ForCausalLM  |    DeepSeek-V2    |   Yes    | Yes  |
|    LlamaForCausalLM       |    Yi             |   Yes    | Yes  |
|    MixtralForCausalLM     |    Mixtral-8x7B   |   Yes    | Yes  |


## 安装
vLLM支持
+ Python 3.8.
+ Python 3.9.
+ Python 3.10.
+ Python 3.11.

### 使用源码编译方式安装

#### 编译环境准备
提供2种环境准备方式：

1. 基于光源pytorch2.1.0基础镜像环境：镜像下载地址：[https://sourcefind.cn/#/image/dcu/pytorch](https://sourcefind.cn/#/image/dcu/pytorch)，根据pytorch2.1.0、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境：安装pytorch2.1.0，pytorch whl包下载目录：[https://cancon.hpccube.com:65024/4/main/pytorch](https://cancon.hpccube.com:65024/4/main/pytorch)，根据python、dtk版本,下载对应pytorch2.1.0的whl包。安装命令如下：
```shell
pip install torch* (下载的torch的whl包)
pip install setuptools wheel
```

#### 源码编译安装
```shell
git clone http://developer.hpccube.com/codes/OpenDAS/vllm.git # 根据需要的分支进行切换
```

- 提供2种源码编译方式（进入vllm目录）：
```
1. 编译whl包并安装
VLLM_INSTALL_PUNICA_KERNELS=1 python setup.py bdist_wheel 
cd dist
pip install vllm*
cd csrc/quantization/gptq
python setup.py bdist_wheel
cd dist
pip install gptq_kernel

2. 源码编译安装
VLLM_INSTALL_PUNICA_KERNELS=1 python3 setup.py install 
```

#### 运行基础环境准备
1、使用上面基于光源pytorch2.1.0基础镜像环境

2、根据pytorch2.1.0、python、dtk及系统下载对应的依赖包：
- triton:[https://cancon.hpccube.com:65024/4/main/triton](https://cancon.hpccube.com:65024/4/main/triton/)
- xformers:[https://cancon.hpccube.com:65024/4/main/xformers](https://cancon.hpccube.com:65024/4/main/xformers)
- flash_attn: [https://cancon.hpccube.com:65024/4/main/flash_attn](https://cancon.hpccube.com:65024/4/main/flash_attn)


#### 注意事项
+ 若使用 pip install 下载安装过慢，可添加源：-i https://pypi.tuna.tsinghua.edu.cn/simple/

## 验证
- python -c "import vllm; print(vllm.\_\_version__)"，版本号与官方版本同步，查询该软件的版本号，例如0.5.0；

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)