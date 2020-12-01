# Dacon: [랜드마크 분류 AI 경진대회](https://dacon.io/competitions/official/235585/overview/)
## 진행 과정
* 기존 데이터셋은 사진들과 json이 들어있는데, 이를 이용해 다음과 같이 각 라벨별로 새로 배치
    ```shell
    dataset/
        class0/
            0.jpg
            1.jpg
            ...
        class1/
            0.jpg
            1.jpg
            ...
        ...
    ```
* 전체 Trainset의 80%를 이용한 학습에서 최고성능을 보인 Epoch을 기록하고, 이후 전체 Trainset을 이용 그 Epoch까지 학습시켜 단일 모델들을 얻음
* Lr scheduler로 StepLR과 CosAnnealing을 사용해 봤는데, 후자가 성능이 더 좋았음
* 모든 경우에서 Cutmix가 성능이 제일 좋음
* Eff_[b0, b1, b5], vit_[base, base_hybrid, large], skresnext50의 모델들을 학습시켜 val loss와 acc를 비교 
    * vit계열은 base를 제외하고 성능이 다 좋지 않았음
    * val loss: b5 > skresnext > b0 > b1 > vit
    * val acc: skresnext > b1 > b5 > b0 > vit
* 앙상블은 단순히 더한것(sum)과 클래스별로 가중치를 주어 더한것(weighted_sum)을 이용. Public Score가 전자가 더 높아 전자를 제출
    * sum: 위에서 비교한 5개의 모델을 모두 이용한 것이, b0를 빼고 나머지 4개를 이용한 것 보다 좋지 않아 최종적으로 b1, b5, vit, skresnext를 사용
    * weighted_sum: 사용한 모델은 sum과 같고, 각 클래스별로 가중치를 주므로 4*1049의 learnable parameter가 추가됨
* 나름 열심히 한 첫 대회. Public 11등 Private 14등

## 사용 방법
```shell
$ python3 main.py -h
```

```console
usage: main.py [-h] --mode {train,val,test} --model {sum,weighted_sum,b0,b1,b2,b3,b4,b5,b6,b7,b8,l2,vit_base,vit_base_hybrid,vit_large,skresnext50} [--calculator False] [--checkpoint None] [--save Checkpoint.pt] [--gpu 0]
               [--cutmix True] [--scheduler {StepLR,Cos}] [--batchsize 128] [--lr 0.001] [--epoch 50] [--flooding 0.01] [--num_classes 1049]

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,val,test}, -m {train,val,test}
                        train: Use total dataset, val: Hold-out of 0.8 ratio, test: Make submission
  --model {sum,weighted_sum,b0,b1,b2,b3,b4,b5,b6,b7,b8,l2,vit_base,vit_base_hybrid,vit_large,skresnext50}
                        sum and weighted_sum: ensemble, others: efficientnet, vit, skresnext
  --calculator False    Caculate mean and std of dataset
  --checkpoint None     Checkpoint directory
  --save Checkpoint.pt  Save directory. If checkpoint exists, save checkpoint in checkpoint dir
  --gpu 0               GPU number to use
  --cutmix True         If True, use Cutmix aug in training
  --scheduler {StepLR,Cos}
                        Cos: cosine annealing
  --batchsize 128
  --lr 0.001
  --epoch 50
  --flooding 0.01
  --num_classes 1049    Number of classes
```