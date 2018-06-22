
# git 설치

```
$ sudo apt-get install git-core  # 기본적인 거는 가능

```


# SSH key

```bash
$ ssh-keygen -t rsa -C "your.email@example.com" -b 4096  #메일을 실제 사용하지 않을거면 적당히  
```
기본값으로 /root/.ssh/  에 생성  

id_rsa.pub  를 등록해주면 된다 

## github 

Setting  -> DeployKey

## gitlab

Setting -> Repository -> Deploy Keys 




