

## 安装环境
```bash
conda create -n simrec python=3.7 -y
conda activate simrec
```
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

## 训练与测试 

```
python train.py --config ./config/config.yaml
```
```
python test.py --eval-weights ./logs/simrec/1/weights/det_best.pth
```
